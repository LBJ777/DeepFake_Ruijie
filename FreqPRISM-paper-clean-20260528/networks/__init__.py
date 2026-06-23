"""FreqPRISM network and detector components."""

__all__ = ["UnifiedArtifactDetector", "UnifiedDetectorConfig"]


def __getattr__(name: str) -> object:
    if name in __all__:
        from networks.detector import UnifiedArtifactDetector, UnifiedDetectorConfig

        return {
            "UnifiedArtifactDetector": UnifiedArtifactDetector,
            "UnifiedDetectorConfig": UnifiedDetectorConfig,
        }[name]
    raise AttributeError(name)
