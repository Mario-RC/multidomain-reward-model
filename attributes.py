"""Shared attribute, domain, and ablation definitions for the pipeline."""

import hashlib

ATTRIBUTES = [
    "co_discourse_structure", "co_logical_consistency", "co_mutual_grounding",
    "co_overall_coherence_score", "co_temporal_causal_coherence", "co_topic_coherence",
    "cs_causality", "cs_coherence", "cs_consistency", "cs_desire", "cs_empathy", "cs_reaction",
    "em_emotional_awareness", "em_emotional_validation", "em_helpful_response",
    "em_overall_empathy_score", "em_perspective_taking", "em_supportive_engagement",
    "mu_coherence", "mu_cultural_specificity", "mu_cultural_value", "mu_empathy", "mu_naturalness",
]

DOMAIN_PREFIXES = {
    "coherence": "co_",
    "commonsense": "cs_",
    "empathy": "em_",
    "multicultural": "mu_",
}

DOMAIN_NAMES = tuple(DOMAIN_PREFIXES)
DOMAIN_TO_INDEX = {name: index for index, name in enumerate(DOMAIN_NAMES)}
DOMAIN_ATTRIBUTE_INDICES = {
    name: tuple(i for i, attribute in enumerate(ATTRIBUTES) if attribute.startswith(prefix))
    for name, prefix in DOMAIN_PREFIXES.items()
}

# Reversible Stage-2 ablations. Stage 1 always predicts the complete
# 23-dimensional vector, so every subset can reuse exactly the same candidate
# embeddings, regression head, examples, and prompt-group split.
ATTRIBUTE_SUBSETS = {
    "full": tuple(ATTRIBUTES),
    # Aggregate labels are derived summaries of their domain subcriteria.
    "no_aggregate": tuple(
        attribute for attribute in ATTRIBUTES
        if attribute not in {"co_overall_coherence_score", "em_overall_empathy_score"}
    ),
    # Remove generic coherence/empathy labels repeated outside their primary
    # domain, as well as the two aggregate labels in the primary domains.
    "domain_specific": tuple(
        attribute for attribute in ATTRIBUTES
        if attribute not in {
            "co_overall_coherence_score", "em_overall_empathy_score",
            "cs_coherence", "cs_empathy", "mu_coherence", "mu_empathy",
        }
    ),
    "no_coherence_overlap": tuple(
        attribute for attribute in ATTRIBUTES
        if attribute not in {
            "co_overall_coherence_score", "cs_coherence", "mu_coherence",
        }
    ),
    "no_empathy_overlap": tuple(
        attribute for attribute in ATTRIBUTES
        if attribute not in {
            "em_overall_empathy_score", "cs_empathy", "mu_empathy",
        }
    ),
}

ATTRIBUTE_SUBSET_CODES = {
    "full": "full",
    "no_aggregate": "na",
    "domain_specific": "ds",
    "no_coherence_overlap": "nco",
    "no_empathy_overlap": "nem",
}


def resolve_active_attributes(subset="full", excluded=None):
    """Resolve and validate a reversible attribute selection.

    Returns (indices, names, excluded_names) in canonical ATTRIBUTES order.
    Every domain must retain at least one dimension because Stage 2 uses
    supervised domain-mass routing.
    """
    if subset not in ATTRIBUTE_SUBSETS:
        raise ValueError(
            f"Unknown attribute subset {subset!r}; choose from "
            f"{sorted(ATTRIBUTE_SUBSETS)}."
        )
    unknown = sorted(set(excluded or ()) - set(ATTRIBUTES))
    if unknown:
        raise ValueError(f"Unknown excluded attribute(s): {unknown}")
    selected = set(ATTRIBUTE_SUBSETS[subset]) - set(excluded or ())
    names = tuple(attribute for attribute in ATTRIBUTES if attribute in selected)
    if not names:
        raise ValueError("Attribute selection removed every dimension.")
    empty_domains = [
        domain for domain, domain_attributes in DOMAIN_ATTRIBUTE_INDICES.items()
        if not any(ATTRIBUTES[index] in selected for index in domain_attributes)
    ]
    if empty_domains:
        raise ValueError(
            "Attribute selection leaves no routing target for domain(s): "
            + ", ".join(empty_domains)
        )
    indices = tuple(ATTRIBUTES.index(attribute) for attribute in names)
    excluded_names = tuple(attribute for attribute in ATTRIBUTES if attribute not in selected)
    return indices, names, excluded_names


def attribute_selection_suffix(subset="full", excluded=None):
    """Return a stable, compact checkpoint suffix for an attribute selection."""
    _, _, excluded_names = resolve_active_attributes(subset, excluded)
    if subset == "full" and not excluded_names:
        return ""
    code = ATTRIBUTE_SUBSET_CODES[subset]
    if excluded:
        digest = hashlib.sha1("\n".join(excluded_names).encode("utf-8")).hexdigest()[:8]
        return f"_as{code}-x{digest}"
    return f"_as{code}"
