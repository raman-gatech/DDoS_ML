
from dataclasses import dataclass

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier

from config import CFG

@dataclass
class DDOSEnsembleBuilder:
    n_estimators: int
    max_depth: int | None
    class_weight: str | None
    apply_pca: bool
    pca_components: int
    scale_before_pca: bool

    def build(self, n_features: int) -> Pipeline:
        steps = []

        if self.scale_before_pca or self.apply_pca:
            steps.append(("scaler", StandardScaler()))

        if self.apply_pca:
            n_comp = min(self.pca_components, n_features)
            steps.append(("pca", PCA(n_components=n_comp, random_state=CFG.random_state)))

        rf = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            class_weight=self.class_weight,
            n_jobs=CFG.n_jobs,
            random_state=CFG.random_state,
        )

        et = ExtraTreesClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            class_weight=self.class_weight,
            n_jobs=CFG.n_jobs,
            random_state=CFG.random_state + 1,
        )

        ensemble = VotingClassifier(
            estimators=[("rf", rf), ("et", et)],
            voting="soft",
            n_jobs=CFG.n_jobs
        )

        steps.append(("clf", ensemble))

        pipeline = Pipeline(steps)
        return pipeline


def build_default_ensemble(n_features: int) -> Pipeline:
    cfg = CFG()
    builder = DDOSEnsembleBuilder(
        n_estimators=cfg.n_estimators,
        max_depth=cfg.max_depth,
        class_weight=cfg.class_weight,
        apply_pca=cfg.apply_pca,
        pca_components=cfg.pca_components,
        scale_before_pca=cfg.scale_before_pca,
    )
    return builder.build(n_features)
