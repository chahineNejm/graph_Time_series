from abc import ABC, abstractmethod
import numpy as np
from typing import Any, Callable, TYPE_CHECKING

from .signal import Port, sem_root

# Use TYPE_CHECKING to prevent circular imports between State and Token
if TYPE_CHECKING:
    from .state import State


class Token(ABC):
    """
    Base class for all actions.
    """
    name: str = "BaseToken"
    token_class: str = "control"
    description: str = "Base description"
    learning_scope: str = "n/a"  # cross_series | within_series | configurable | n/a
    
    # Caps and limits (Default to infinite, override later in config/algo)
    class_max_uses: int | float = float('inf')  
    max_uses: int | float = float('inf')        
    
    # Metadata for dependency checks, logging, and graph construction.
    # Can be a dict, list, tuple, or set of state keys.
    reads: dict | list | tuple | set = ()
    writes: dict | list | tuple | set = ()

    # Declarative port contract over the Signal board. When non-empty, these
    # supersede the string-based reads/writes for dependency checking: a token
    # is applicable only when every required (non-optional) port is satisfiable
    # by meaning, optionally via adapter coercion. Tokens that still use the
    # legacy reads/writes keep working because these default to empty.
    requires: tuple = ()
    provides: tuple = ()

    def can_apply(self, state: 'State') -> bool:
        """
        Universal verification logic. 
        Checks caps before delegating to the specific logical conditions.
        """
        # 1. Check global category cap 
        if state.class_counts.get(self.token_class, 0) >= self.class_max_uses:
            return False
            
        # 2. Check specific token cap 
        uses = state.token_sequence.count(self.name)
        if uses >= self.max_uses:
            return False

        # 3. Check declared state dependencies (legacy string reads)
        if not self.dependencies_available(state):
            return False

        # 3b. Check declarative port requirements (Signal board)
        if self.requires and not self.ports_satisfied(state):
            return False

        # 4. Check token-specific business logic
        return self.check_specific_conditions(state)

    def ports_satisfied(self, state: 'State') -> bool:
        """True iff every required, non-optional port is satisfiable."""
        for port in self.requires:
            if getattr(port, "optional", False):
                continue
            if not self._port_ok(state, port):
                return False
        return True

    def _port_ok(self, state: 'State', port: 'Port') -> bool:
        space = port.space
        if isinstance(space, str) and space not in {"current", "raw", "any"}:
            space = "any"
        alignment = None if port.alignment == "any" else port.alignment
        matches = state.query(
            sem=port.sem,
            axes=port.axes,
            alignment=alignment,
            space=space,
            tags=(port.tags or None),
        )
        if matches:
            return True
        # Adapter fallback: a features port can be met by anything coercible.
        if getattr(port, "coerce", False) and sem_root(port.sem) == "features":
            try:
                bundle = state.feature_bundle(
                    select_tags=frozenset(port.tags or ()),
                )
                return bundle.matrix.shape[1] > 0
            except Exception:
                return False
        return False

    def dependencies_available(self, state: 'State') -> bool:
        """Check that all declared reads exist somewhere in State."""
        available_stores = (
            getattr(state, "historical_features", {}),
            getattr(state, "future_features", {}),
            getattr(state, "features", {}),
            getattr(state, "metadata", {}),
            getattr(state, "flags", {}),
        )
        return all(
            any(key in store for store in available_stores)
            for key in self._dependency_keys(self.reads)
        )

    @staticmethod
    def _dependency_keys(values) -> tuple:
        if values is None:
            return ()
        if isinstance(values, dict):
            return tuple(values.keys())
        if isinstance(values, str):
            return (values,)
        return tuple(values)

    def check_specific_conditions(self, state: 'State') -> bool:
        """
        Safe default hook. Concrete tokens can override this when they need
        extra conditions beyond caps and declared dependencies.
        """
        return True

    @abstractmethod
    def apply(self, state: 'State') -> 'State':
        """Core execution block."""
        pass

    def _log_execution(self, state: 'State', reads: dict = None, writes: dict = None):
        """Standardized state mutation tracking."""
        state.record_token(
            self.name,
            self.token_class,
            self._metadata_dict(reads if reads is not None else self.reads),
            self._metadata_dict(writes if writes is not None else self.writes),
        )

    @staticmethod
    def _metadata_dict(values) -> dict:
        if values is None:
            return {}
        if isinstance(values, dict):
            return values
        if isinstance(values, str):
            return {values: None}
        return {key: None for key in values}


# ==========================================
# INTERMEDIATE BASE CLASSES
# ==========================================

class CleaningToken(Token):
    """Base for imputation, outlier removal, or base scaling."""
    token_class = "cleaning"

    def check_specific_conditions(self, state: 'State') -> bool:
        # Standard constraint: Clean data before training models
        if state.n_models_applied > 0:
            return False
        return super().check_specific_conditions(state)
    

class FeatureToken(Token):
    """Base for generating features."""
    token_class = "feature"

    @abstractmethod
    def apply(self, state: 'State') -> 'State':
        """
        Concrete FeatureTokens must implement this to extract data from the state,
        create features, and place them back into state dictionaries as they see fit.
        Make sure to call `self._log_execution(state)` at the end.
        """
        pass


class TransformToken(Token):
    """Base for mutating the state's target or features (requires inverse mapping)."""
    token_class = "transform"

    def check_specific_conditions(self, state: 'State') -> bool:
        # Standard constraint: Transforms generally must happen before model training
        if state.n_models_applied > 0:
            return False
        return super().check_specific_conditions(state)

    def get_inverse_fn(self) -> Callable[[np.ndarray], np.ndarray] | None:
        """
        Optional hook for transforms that expose their inverse separately.
        Most transforms can register directly with state.register_transform().
        """
        return None

    @abstractmethod
    def apply(self, state: 'State') -> 'State':
        """
        Mutate the state, call `state.register_transform()` when needed,
        and call `self._log_execution()`.
        """
        pass


class ModelToken(Token):
    """Base for predictive modeling using the residual stack."""
    token_class = "model"

    def check_specific_conditions(self, state: 'State') -> bool:
        # Standard constraint: Cannot train a model without any features
        if not state.historical_features:
            return False
        return super().check_specific_conditions(state)

    @abstractmethod
    def get_model(self) -> Any:
        """Return the uninstantiated or instantiated model object."""
        pass

    @abstractmethod
    def apply(self, state: 'State') -> 'State':
        """
        Concrete ModelTokens must pull features/targets from the state, train,
        generate predictions, call `state.push_prediction()`, and call `self._log_execution()`.
        """
        pass


class ControlToken(Token):
    """System tokens like START, STOP."""
    token_class = "control"
    
    def apply(self, state: 'State') -> 'State':
        if self.name == "STOP":
            state.terminated = True
        self._log_execution(state)
        return state
