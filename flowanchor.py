"""
FlowAnchor: Stabilizing the Editing Signal for Inversion-Free Video Editing
Core implementation of SAR (Spatial-aware Attention Refinement) and AMM (Adaptive Magnitude Modulation)

Enhanced with multi-subject support:
- Three-category SAR (target / preserve / other) to prevent editing leakage
- Subject-isolated AMM with per-region contrast normalization
- T5 subword-aware token matching
"""
import torch
from typing import Optional, List, Dict, Tuple


class FlowAnchorEditor:
    """
    FlowAnchor editing framework that stabilizes the editing signal
    via SAR and AMM on top of Wan-Edit (FlowEdit for videos).

    Enhanced for multi-subject scenarios with:
    - Preservation anchors to protect unedited subjects
    - Per-subject attention refinement
    - Subject-isolated magnitude modulation
    """

    def __init__(
        self,
        device: torch.device,
        beta1: float = 0.5,
        beta2: float = 0.5,
        gamma_scale: float = 1.0,
        beta_preserve: float = 0.3,   # preservation suppression strength
        beta_cross: float = 0.2,      # cross-subject leakage suppression
    ):
        self.device = device
        self.beta1 = beta1
        self.beta2 = beta2
        self.gamma_scale = gamma_scale
        self.beta_preserve = beta_preserve
        self.beta_cross = beta_cross

    @staticmethod
    def find_target_token_indices(
        prompt: str,
        target_words: List[str],
        tokenizer=None,
    ) -> List[int]:
        """
        Find token indices for target words in the prompt.
        
        Uses T5 tokenizer if available for accurate subword matching,
        otherwise falls back to whitespace splitting.
        """
        if tokenizer is not None:
            return FlowAnchorEditor._find_indices_with_tokenizer(
                prompt, target_words, tokenizer)
        return FlowAnchorEditor._find_indices_whitespace(prompt, target_words)

    @staticmethod
    def _find_indices_whitespace(prompt: str, target_words: List[str]) -> List[int]:
        """Simple whitespace-based token index lookup."""
        tokens = prompt.lower().split()
        if not tokens:
            return []
        indices = []
        for word in target_words:
            word_lower = word.lower()
            for i, t in enumerate(tokens):
                if word_lower in t and i not in indices:
                    indices.append(i)
        return indices if indices else list(range(len(tokens)))

    @staticmethod
    def _find_indices_with_tokenizer(
        prompt: str,
        target_words: List[str],
        tokenizer,
    ) -> List[int]:
        """
        T5 subword-aware token index matching.
        
        Handles cases like "sweater" → ["s", "weater"] in T5 tokenization
        by encoding each target word and matching encoded token IDs.
        """
        # Get full prompt tokenization
        full_encoding = tokenizer(
            prompt,
            return_tensors=None,
            add_special_tokens=True,
        )
        prompt_token_ids = full_encoding.input_ids if hasattr(
            full_encoding, 'input_ids') else full_encoding['input_ids']

        indices = []
        for word in target_words:
            # Encode the target word in isolation
            word_encoding = tokenizer(
                word,
                return_tensors=None,
                add_special_tokens=False,
            )
            word_ids = word_encoding.input_ids if hasattr(
                word_encoding, 'input_ids') else word_encoding['input_ids']

            # Find all occurrences in the prompt (sliding window)
            wlen = len(word_ids)
            for i in range(len(prompt_token_ids) - wlen + 1):
                if prompt_token_ids[i:i + wlen] == word_ids:
                    for j in range(i, i + wlen):
                        if j not in indices:
                            indices.append(j)
        return indices if indices else list(range(len(prompt_token_ids)))

    @staticmethod
    def parse_prompt_phrases(prompt: str) -> Dict[str, List[str]]:
        """
        Parse a prompt into subject phrases for multi-subject editing.
        
        Returns a dict with:
        - 'subjects': list of noun phrases (e.g., ["a woman in a pink sweater", 
                       "a person in a gray top"])
        - 'attributes': per-subject attribute words for color/material changes
        
        Uses simple heuristic parsing based on common video editing patterns.
        """
        import re
        
        result = {'subjects': [], 'attributes': []}
        
        # Split on common subject separators
        separators = [
            r'\b(?:and|,)\s+(?=a\s|the\s|an\s)',  # "and a", ", a"
            r'\bwalks?\s+(?:forward|toward|to)\b',  # action verbs
            r'\bhugs?\b',
            r'\bstanding\s+next\s+to\b',
            r'\bsitting\s+(?:next\s+to|beside)\b',
        ]
        
        # Try to identify subject phrases
        # Pattern: "a/an/the [descriptor]* [noun]" 
        subject_pattern = re.compile(
            r'(?:a|an|the)\s+(?:[\w-]+\s+){0,5}?(?:'
            r'person|woman|man|child|dog|cat|car|bus|bike'
            r'|people|girl|boy|animal'
            r')[\w\s,.-]*',
            re.IGNORECASE
        )
        
        subjects = subject_pattern.findall(prompt)
        result['subjects'] = [s.strip(',. ') for s in subjects]
        
        # Extract color/material attributes per subject
        color_pattern = re.compile(
            r'\b(red|blue|green|yellow|pink|purple|orange|black|white|'
            r'gray|grey|brown|lemon|navy|teal|maroon|gold|silver|beige|'
            r'violet|indigo|cyan|magenta|coral|turquoise|lavender|'
            r'crimson|scarlet|amber|jade|ruby|sapphire|emerald)\b',
            re.IGNORECASE
        )
        
        for subj in subjects:
            colors = color_pattern.findall(subj)
            result['attributes'].append(colors)
        
        return result

    def _prepare_mask(self, mask: torch.Tensor, ca_map: torch.Tensor
                      ) -> torch.Tensor:
        """Normalize mask to [B, L_v] shape for attention map operations."""
        B, L_v, L_t = ca_map.shape
        mask = mask.to(ca_map.device)
        if mask.dim() == 5:
            mask_flat = mask.reshape(B, -1)
        elif mask.dim() == 2:
            mask_flat = mask
        else:
            mask_flat = mask.reshape(B, -1)

        if mask_flat.shape[1] < L_v:
            pad = torch.zeros(B, L_v - mask_flat.shape[1],
                              device=ca_map.device, dtype=mask_flat.dtype)
            mask_flat = torch.cat([mask_flat, pad], dim=1)
        else:
            mask_flat = mask_flat[:, :L_v]
        return mask_flat > 0.5

    def _build_token_category_mask(
        self,
        B: int, L_v: int, L_t: int,
        target_indices: List[int],
        preserve_indices: List[int],
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Build boolean masks for three token categories:
        - is_target: tokens to edit
        - is_preserve: tokens to protect (other subjects' attributes)
        - is_other: all remaining tokens
        """
        is_target = torch.zeros(B, L_v, L_t, device=device, dtype=torch.bool)
        is_preserve = torch.zeros(B, L_v, L_t, device=device, dtype=torch.bool)

        if target_indices:
            t_idx = torch.tensor(target_indices, device=device, dtype=torch.long)
            idx = t_idx.unsqueeze(0).unsqueeze(0).expand(B, L_v, -1)
            is_target.scatter_(2, idx, True)

        if preserve_indices:
            p_idx = torch.tensor(preserve_indices, device=device, dtype=torch.long)
            idx_p = p_idx.unsqueeze(0).unsqueeze(0).expand(B, L_v, -1)
            is_preserve.scatter_(2, idx_p, True)
            # Prevent overlap: preserve takes priority over target
            is_target = is_target & ~is_preserve

        is_other = ~(is_target | is_preserve)
        return is_target, is_preserve, is_other

    def spatial_aware_attention_refinement(
        self,
        ca_map: torch.Tensor,
        mask: torch.Tensor,
        target_indices: List[int],
        preserve_indices: Optional[List[int]] = None,
    ) -> torch.Tensor:
        """
        Three-category SAR for multi-subject editing.
        
        Categories:
        - target tokens:  boosted inside mask, suppressed outside
        - preserve tokens: protected inside mask (mitigates leakage to other subjects)
        - other tokens:    standard suppression inside mask
        
        This prevents editing operations from affecting unedited subjects
        in multi-person / multi-object scenes.
        """
        B, L_v, L_t = ca_map.shape

        mask_bool = self._prepare_mask(mask, ca_map)
        mask_expanded = mask_bool.unsqueeze(-1)  # [B, L_v, 1]

        preserve_indices = preserve_indices or []
        is_target, is_preserve, is_other = self._build_token_category_mask(
            B, L_v, L_t, target_indices, preserve_indices, ca_map.device)

        A = ca_map.clone()

        # ---- Step 1: Per-token modulation within masked region ----
        A_max = A.max(dim=-1, keepdim=True).values   # [B, L_v, 1]
        A_min = A.min(dim=-1, keepdim=True).values   # [B, L_v, 1]

        boost_target = self.beta1 * (A_max - A)       # boost target tokens
        suppress_other = self.beta1 * (A - A_min)     # suppress non-target tokens
        suppress_preserve = self.beta_preserve * (A - A_min)  # milder suppression

        target_in_mask = mask_expanded & is_target
        preserve_in_mask = mask_expanded & is_preserve
        other_in_mask = mask_expanded & is_other

        step1 = A.clone()
        step1 = torch.where(target_in_mask, A + boost_target, step1)
        step1 = torch.where(other_in_mask, A - suppress_other, step1)
        step1 = torch.where(preserve_in_mask, A - suppress_preserve, step1)

        # ---- Step 2: Spatial competition (in-mask vs out-of-mask) ----
        A_prime_max = step1.max(dim=1, keepdim=True).values  # [B, 1, 1]
        A_prime_min = step1.min(dim=1, keepdim=True).values  # [B, 1, 1]

        boost2 = self.beta2 * (A_prime_max - step1)          # reinforce in-mask
        suppress2 = self.beta2 * (step1 - A_prime_min)       # suppress out-of-mask
        suppress_cross = self.beta_cross * (step1 - A_prime_min)  # cross-subject

        in_mask_target = mask_expanded & is_target       # target inside mask
        out_mask_target = ~mask_expanded & is_target     # target outside mask (leakage)

        step2 = step1.clone()
        step2 = torch.where(in_mask_target, step1 + boost2, step2)
        step2 = torch.where(out_mask_target, step1 - suppress2, step2)
        # NOTE: preserve tokens outside mask are left unchanged —
        # they represent other subjects that should maintain natural attention

        return step2

    def adaptive_magnitude_modulation(
        self,
        delta_v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        num_frames: int = 1,
        per_region: bool = False,
    ) -> torch.Tensor:
        """
        Adaptive Magnitude Modulation with optional per-region normalization.

        When per_region=True (recommended for multi-subject scenes):
        - Computes contrast statistics independently within masked vs. unmasked regions
        - Prevents high-contrast regions from dominating modulation
        - Each subject's editing signal is normalized against its own context
        
        Args:
            delta_v: editing direction [B, C, F, H, W]
            mask: spatial mask, same shape as delta_v or broadcastable
            num_frames: number of frames for frame-aware scaling
            per_region: if True, normalize contrast independently per region
        """
        if mask is not None:
            # Ensure mask is on same device and broadcastable
            mask = mask.to(delta_v.device)
            while mask.dim() < delta_v.dim():
                mask = mask.unsqueeze(0)
            # Expand to match delta_v shape
            if mask.shape[1] == 1 and delta_v.shape[1] > 1:
                mask = mask.expand(-1, delta_v.shape[1], -1, -1, -1)
            masked_signal = delta_v * mask
        else:
            masked_signal = delta_v

        abs_signal = torch.abs(masked_signal)

        if per_region and mask is not None:
            # ---- Per-region contrast normalization ----
            # Compute stats independently inside and outside mask
            mask_bool = mask > 0.5
            eps = 1e-8

            # Inside masked region
            in_mask = mask_bool.float()
            in_count = in_mask.sum(dim=(-3, -2, -1), keepdim=True) + eps
            in_mean = (abs_signal * in_mask).sum(dim=(-3, -2, -1), keepdim=True) / in_count
            in_var = ((abs_signal - in_mean) ** 2 * in_mask).sum(
                dim=(-3, -2, -1), keepdim=True) / in_count
            in_std = torch.sqrt(in_var + eps)

            # Outside masked region
            out_mask = (1 - in_mask)
            out_count = out_mask.sum(dim=(-3, -2, -1), keepdim=True) + eps
            out_mean = (abs_signal * out_mask).sum(dim=(-3, -2, -1), keepdim=True) / out_count

            # Per-region contrast: use region-specific statistics
            contrast_map = torch.zeros_like(abs_signal)
            contrast_map = torch.where(
                mask_bool,
                (abs_signal - in_mean) / in_std,
                (abs_signal - out_mean) / (abs_signal.std(dim=(-3, -2, -1), keepdim=True) + eps),
            )
        else:
            # ---- Original global contrast (unchanged for single-subject) ----
            spatial_mean = abs_signal.mean(dim=(-3, -2, -1), keepdim=True)
            spatial_std = abs_signal.std(dim=(-3, -2, -1), keepdim=True) + 1e-8
            contrast_map = (abs_signal - spatial_mean) / spatial_std

        contrast_map = torch.clamp(contrast_map, min=0)
        contrast_sum = contrast_map.sum(dim=(-3, -2, -1), keepdim=True) + 1e-8
        contrast_map = contrast_map / contrast_sum

        if mask is not None:
            contrast_map = contrast_map * mask

        frame_scale = self.gamma_scale * (num_frames / 81.0) ** 0.5

        amplified = delta_v + frame_scale * (contrast_map * delta_v)

        return amplified


class MultiSubjectFlowAnchorEditor(FlowAnchorEditor):
    """
    Enhanced FlowAnchor editor for multi-person / multi-object scenes.

    Key additions over the base FlowAnchorEditor:
    1. Automatic subject parsing from natural language prompts
    2. Preservation anchor extraction (tokens for unedited subjects)
    3. Convenience method for complete multi-subject SAR + AMM pipeline
    4. Delta-V decomposition to isolate per-subject editing signals

    Usage example:
        editor = MultiSubjectFlowAnchorEditor(device)
        subject_info = editor.analyze_multi_subject_prompt(
            "A woman in a pink sweater hugs a person in a gray top",
            "A woman in a lemon sweater hugs a person in a gray top"
        )
        # subject_info['target_indices'] → tokens for "lemon sweater"
        # subject_info['preserve_indices'] → tokens for "gray top"
    """

    # Common attribute words (colors, materials, patterns) that
    # are frequently edited in video editing tasks
    ATTRIBUTE_PATTERNS = [
        # Colors
        r'\b(red|blue|green|yellow|pink|purple|orange|black|white|'
        r'gray|grey|brown|lemon|navy|teal|maroon|gold|silver|beige|'
        r'violet|indigo|cyan|magenta|coral|turquoise|lavender|'
        r'crimson|scarlet|amber|jade|ruby|sapphire|emerald|mint|'
        r'peach|burgundy|mustard|olive|plum|rose|tan|ivory|charcoal)\b',
        # Materials/textures
        r'\b(leather|denim|silk|cotton|wool|velvet|linen|lace|suede|'
        r'knit|woven|patterned|striped|plaid|floral|polka|checkered|'
        r'glitter|metallic|matte|glossy|shiny|sparkling|transparent)\b',
        # Clothing items
        r'\b(sweater|jacket|coat|shirt|top|dress|skirt|pants|jeans|'
        r'shorts|hoodie|blouse|t-shirt|tank|cardigan|blazer|vest|'
        r'scarf|hat|cap|shoes|boots|sneakers|heels|sandals)\b',
    ]

    # Common subject head nouns
    SUBJECT_HEAD_NOUNS = [
        'person', 'woman', 'man', 'child', 'girl', 'boy', 'baby',
        'people', 'dog', 'cat', 'bird', 'horse', 'car', 'bus', 'bike',
        'motorcycle', 'truck', 'boat', 'airplane',
    ]

    def analyze_multi_subject_prompt(
        self,
        src_prompt: str,
        tgt_prompt: str,
        tokenizer=None,
    ) -> Dict:
        """
        Analyze source and target prompts for multi-subject editing.

        Returns a dict with:
        - 'target_words': words that differ between src and tgt
        - 'preserve_words': words in src that should stay unchanged
        - 'src_subjects': parsed subject phrases from source
        - 'tgt_subjects': parsed subject phrases from target
        - 'is_multi_subject': whether the scene has multiple subjects
        """
        import re
        import difflib

        result = {
            'target_words': [],
            'preserve_words': [],
            'src_subjects': [],
            'tgt_subjects': [],
            'is_multi_subject': False,
        }

        # ---- 1. Detect changed words using diff ----
        src_words = src_prompt.lower().split()
        tgt_words = tgt_prompt.lower().split()

        matcher = difflib.SequenceMatcher(None, src_words, tgt_words)
        changed_src = []
        changed_tgt = []

        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'replace':
                changed_src.extend(src_words[i1:i2])
                changed_tgt.extend(tgt_words[j1:j2])
            elif tag == 'delete':
                changed_src.extend(src_words[i1:i2])
            elif tag == 'insert':
                changed_tgt.extend(tgt_words[j1:j2])

        # Filter to meaningful attribute words
        attr_re = re.compile('|'.join(self.ATTRIBUTE_PATTERNS), re.IGNORECASE)

        result['target_words'] = [
            w for w in changed_tgt if attr_re.match(w)
        ] or changed_tgt  # fallback to all changed words

        # Preservation words: attribute words in src that did NOT change
        all_src_attrs = set(
            w for w in src_words if attr_re.match(w))
        changed_src_attrs = set(
            w for w in changed_src if attr_re.match(w))
        result['preserve_words'] = list(
            all_src_attrs - changed_src_attrs)

        # ---- 2. Parse subject phrases ----
        result['src_subjects'] = self._extract_subject_phrases(src_prompt)
        result['tgt_subjects'] = self._extract_subject_phrases(tgt_prompt)

        # ---- 3. Determine if multi-subject ----
        result['is_multi_subject'] = (
            len(result['src_subjects']) > 1 or
            len(result['tgt_subjects']) > 1
        )

        return result

    def _extract_subject_phrases(self, prompt: str) -> List[str]:
        """Extract subject noun phrases from a prompt."""
        import re

        subjects = []
        # Pattern: "a/an/the [modifiers]* [head_noun]"
        head_nouns_alt = '|'.join(self.SUBJECT_HEAD_NOUNS)
        pattern = re.compile(
            rf'(?:a|an|the)\s+(?:[\w-]+\s+){{0,6}}?(?:{head_nouns_alt})\b',
            re.IGNORECASE
        )
        matches = pattern.findall(prompt)
        subjects = [m.strip(',. ') for m in matches]
        return subjects

    def get_edit_anchors(
        self,
        src_prompt: str,
        tgt_prompt: str,
        tokenizer=None,
    ) -> Tuple[List[int], List[int]]:
        """
        Get target and preserve token indices for multi-subject editing.

        Returns:
            (target_indices, preserve_indices) for use with SAR.
        
        For "pink sweater → lemon sweater" with "gray top" unchanged:
            target_indices: [lemon, sweater]
            preserve_indices: [gray, top]
        """
        analysis = self.analyze_multi_subject_prompt(
            src_prompt, tgt_prompt, tokenizer)

        target_indices = self.find_target_token_indices(
            tgt_prompt, analysis['target_words'], tokenizer)

        preserve_indices = []
        if analysis['preserve_words']:
            preserve_indices = self.find_target_token_indices(
                src_prompt, analysis['preserve_words'], tokenizer)

        return target_indices, preserve_indices

    def multi_subject_sar_amm(
        self,
        ca_map: torch.Tensor,
        mask: torch.Tensor,
        delta_v: torch.Tensor,
        src_prompt: str,
        tgt_prompt: str,
        num_frames: int = 1,
        tokenizer=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Complete multi-subject pipeline: SAR + AMM with auto anchor detection.

        Combines:
        1. Automatic detection of target vs. preserve words from prompts
        2. Three-category SAR with preservation anchors
        3. Per-region AMM for subject-isolated modulation

        Returns:
            (refined_ca_map, amplified_delta_v)
        """
        target_indices, preserve_indices = self.get_edit_anchors(
            src_prompt, tgt_prompt, tokenizer)

        refined_ca = self.spatial_aware_attention_refinement(
            ca_map, mask,
            target_indices=target_indices,
            preserve_indices=preserve_indices,
        )

        amplified_delta = self.adaptive_magnitude_modulation(
            delta_v, mask=mask,
            num_frames=num_frames,
            per_region=True,
        )

        return refined_ca, amplified_delta
