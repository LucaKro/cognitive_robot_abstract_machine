from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from typing_extensions import Generic, Optional, Type, Dict, Any, List, Union, Self, Iterable

from krrood.entity_query_language.symbolic import Exists, ResultQuantifier, An, UnificationDict, DomainType

from .entity import (
    ConditionType,
    contains,
    in_,
    flatten,
    let,
    set_of,
    entity,
    exists,
)
from .failures import NoneWrappedFieldError
from .predicate import HasType
from .symbolic import (
    CanBehaveLikeAVariable,
    Attribute,
    Comparator,
    Flatten,
    QueryObjectDescriptor,
    Selectable,
    SymbolicExpression,
    OperationResult,
    Literal,
    SetOf,
    Entity,
)
from .utils import is_iterable, T


@dataclass
class Quantifier:
    """
    A class representing a quantifier in a Match statement. This is used to quantify the result of the match.
    """
    type_: Type[ResultQuantifier]
    """
    The type of the quantifier.
    """
    kwargs: Dict[str, Any] = field(default_factory=dict)
    """
    The keyword arguments to pass to the quantifier.
    """

    def apply(self, expr: QueryObjectDescriptor) -> Union[ResultQuantifier[T], T]:
        return self.type_(_child_=expr, **self.kwargs)


@dataclass
class Match(Selectable[T]):
    """
    Construct a query that looks for the pattern provided by the type and the keyword arguments.
    Example usage where we look for an object of type Drawer with body of type Body that has the name"drawer_1":
        >>> @dataclass
        >>> class Body:
        >>>     name: str
        >>> @dataclass
        >>> class Drawer:
        >>>     body: Body
        >>> drawer = matching(Drawer)(body=matching(Body)(name="drawer_1"))
    """

    _type_: Optional[Type[T]] = None
    """
    The type of the variable.
    """
    _domain_: DomainType = field(default=None, kw_only=True)
    """
    The domain to use for the variable created by the match.
    """
    _kwargs_: Dict[str, Any] = field(init=False, default_factory=dict)
    """
    The keyword arguments to match against.
    """
    _variable_: Optional[CanBehaveLikeAVariable[T]] = field(kw_only=True, default=None)
    """
    The created variable from the type and kwargs.
    """
    _conditions_: List[ConditionType] = field(init=False, default_factory=list)
    """
    The conditions that define the match.
    """
    _selected_variables_: List[CanBehaveLikeAVariable] = field(
        init=False, default_factory=list
    )
    """
    A list of selected attributes.
    """
    _parent_match_: Optional[Match] = field(init=False, default=None)
    """
    The parent match if this is a nested match.
    """
    _is_selected_: bool = field(default=False, kw_only=True)
    """
    Whether the variable should be selected in the result.
    """
    _existential_: bool = field(default=False, kw_only=True)
    """
    Whether the match is an existential match check or not.
    """
    _universal_: bool = field(default=False, kw_only=True)
    """
    Whether the match is a universal match (i.e., must match for all values of the variable/attribute) check or not.
    """
    _attributes_: Dict[str, AttributeAssignment] = field(init=False, default_factory=dict)
    """
    A dictionary mapping attribute names to their corresponding attribute assignments.
    """
    _quantifier_data_: Optional[Quantifier] = field(init=False, default_factory=lambda: Quantifier(An))
    """
    The quantifier data for the match.
    """

    def __post_init__(self):
        """
        This is needed to prevent the SymbolicExpression __post_init__ from being called which will make a node out of
        this instance, and that is not what we want.
        """
        ...

    def __call__(self, **kwargs) -> Union[Self, T, CanBehaveLikeAVariable[T]]:
        """
        Update the match with new keyword arguments to constrain the type we are matching with.

        :param kwargs: The keyword arguments to match against.
        :return: The current match instance after updating it with the new keyword arguments.
        """
        self._kwargs_ = kwargs
        return self

    def _resolve_(
        self,
        variable: Optional[CanBehaveLikeAVariable] = None,
        parent: Optional[Match] = None,
    ):
        """
        Resolve the match by creating the variable and conditions expressions.

        :param variable: An optional pre-existing variable to use for the match; if not provided, a new variable will
         be created.
        :param parent: The parent match if this is a nested match.
        :return:
        """
        self._update_fields_(variable, parent)
        for attr_name, attr_assigned_value in self._kwargs_.items():
            attr_assignment = AttributeAssignment(
                attr_name, self._variable_, attr_assigned_value
            )
            self._attributes_[attr_name] = attr_assignment
            if isinstance(attr_assigned_value, Select):
                self._update_selected_variables_(attr_assignment.attr)
                attr_assigned_value._var_ = attr_assignment.attr
            if attr_assignment.is_an_unresolved_match:
                attr_assignment.resolve(self)
                self._conditions_.extend(attr_assignment.conditions)
            else:
                condition = (
                    attr_assignment.infer_condition_between_attribute_and_assigned_value()
                )
                self._conditions_.append(condition)
        return self

    def _set_as_selected_(self):
        self._is_selected_ = True
        self._update_selected_variables_(self._variable_)

    def _evaluate__(
            self,
            sources: Optional[Dict[int, HashedValue]] = None,
            parent: Optional[SymbolicExpression] = None,
    ) -> Iterable[OperationResult]:
        yield from self._variable_._evaluate__(sources, parent)

    @property
    def _name_(self) -> str:
        return self._var_._name_

    @cached_property
    def _all_variable_instances_(self) -> List[CanBehaveLikeAVariable[T]]:
        return self._var_._all_variable_instances_

    def _update_fields_(
        self,
        variable: Optional[CanBehaveLikeAVariable] = None,
        parent: Optional[Match] = None,
    ):
        """
        Update the match variable, parent, is_selected, and type_ fields.

        :param variable: The variable to use for the match.
         If None, a new variable will be created.
        :param parent: The parent match if this is a nested match.
        """

        if variable is not None:
            self._variable_ = variable
        elif self._variable_ is None:
            self._variable_ = let(self._type_, self._domain_)

        if not self._var_:
            self._var_ = variable

        self._parent_match_ = parent

        if self._is_selected_:
            self._update_selected_variables_(self._variable_)

        if not self._type_:
            self._type_ = self._variable_._type_

    def _update_selected_variables_(self, variable: CanBehaveLikeAVariable):
        """
        Update the selected variables of the match by adding the given variable to the root Match selected variables.
        """
        if hash(variable) not in map(hash, self._root_match_._selected_variables_):
            self._root_match_._selected_variables_.append(variable)

    @property
    def _root_match_(self) -> Match:
        if self._parent_match_:
            return self._parent_match_._root_match_
        return self

    @cached_property
    def _expression_(self) -> Union[ResultQuantifier[T], T]:
        """
        Return the entity expression corresponding to the match query.
        """
        if not self._variable_:
            self._resolve_()
        if len(self._selected_variables_) > 1:
            query_descriptor = set_of(self._selected_variables_, *self._conditions_)
        else:
            if not self._selected_variables_:
                self._selected_variables_.append(self._variable_)
            query_descriptor = entity(self._selected_variables_[0], *self._conditions_)
        return self._quantifier_data_.apply(query_descriptor)

    def domain_from(self, domain: DomainType):
        """
        Record the domain to use for the variable created by the match.
        """
        self._domain_ = domain
        return self

    def _quantify_(
            self, quantifier: Type[ResultQuantifier], **quantifier_kwargs
    ) -> Union[ResultQuantifier[T], T]:
        """
        Record the quantifier to be applied to the result of the match.
        """
        self._quantifier_data_ = Quantifier(quantifier, quantifier_kwargs)
        return self

    def evaluate(self):
        """
        Evaluate the match expression and return the result.
        """
        return self._expression_.evaluate()

    def __getattr__(self, item):
        return self._attributes_[item].assigned_value


@dataclass
class AttributeAssignment:
    """
    A class representing an attribute assignment in a Match statement.
    """

    attr_name: str
    """
    The name of the attribute to assign the value to.
    """
    variable: CanBehaveLikeAVariable
    """
    The variable whose attribute is being assigned.
    """
    assigned_value: Union[Literal, Match]
    """
    The value to assign to the attribute, which can be a Match instance or a Literal.
    """
    conditions: List[ConditionType] = field(init=False, default_factory=list)
    """
    The conditions that define attribute assignment.
    """

    def resolve(self, parent_match: Match):
        """
        Resolve the attribute assignment by creating the conditions and applying the necessary mappings
        to the attribute.

        :param parent_match: The parent match of the attribute assignment.
        """
        possibly_flattened_attr = self.attr
        if self.attr._is_iterable_ and (
                self.assigned_value._kwargs_ or self.is_type_filter_needed
        ):
            possibly_flattened_attr = flatten(self.attr)

        self.assigned_value._resolve_(possibly_flattened_attr, parent_match)

        if self.is_type_filter_needed:
            self.conditions.append(
                HasType(possibly_flattened_attr, self.assigned_value._type_)
            )

        self.conditions.extend(self.assigned_value._conditions_)

    def infer_condition_between_attribute_and_assigned_value(
        self,
    ) -> Union[Comparator, Exists]:
        """
        Find and return the appropriate condition for the attribute and its assigned value. This can be one of contains,
        in_, or == depending on the type of the assigned value and the type of the attribute. In addition, if the
        assigned value is a Match instance with an existential flag set, an Exists expression is created over the
         comparator condition.

        :return: A Comparator or an Exists expression representing the condition.
        """
        if self.attr._is_iterable_ and not self.is_iterable_value:
            condition = contains(self.attr, self.assigned_variable)
        elif not self.attr._is_iterable_ and self.is_iterable_value:
            condition = in_(self.attr, self.assigned_variable)
        elif (
            self.attr._is_iterable_
            and self.is_iterable_value
            and not (
                isinstance(self.assigned_value, Match) and self.assigned_value._universal_
            )
        ):
            condition = contains(self.assigned_variable, flatten(self.attr))
        else:
            condition = self.attr == self.assigned_variable

        if isinstance(self.assigned_value, Match) and self.assigned_value._existential_:
            condition = exists(self.attr, condition)

        return condition

    @cached_property
    def assigned_variable(self) -> CanBehaveLikeAVariable:
        """
        :return: The symbolic variable representing the assigned value.
        """
        return (
            self.assigned_value._variable_
            if isinstance(self.assigned_value, Match)
            else self.assigned_value
        )

    @cached_property
    def attr(self) -> Attribute:
        """
        :return: the attribute of the variable.
        :raises NoneWrappedFieldError: If the attribute does not have a WrappedField.
        """
        attr: Attribute = getattr(self.variable, self.attr_name)
        if not attr._wrapped_field_:
            raise NoneWrappedFieldError(self.variable._type_, self.attr_name)
        return attr

    @property
    def is_an_unresolved_match(self) -> bool:
        """
        :return: True if the value is an unresolved Match instance, else False.
        """
        return (
            isinstance(self.assigned_value, Match) and not self.assigned_value._variable_
        )

    @cached_property
    def is_iterable_value(self) -> bool:
        """
        :return: True if the value is an iterable or a Match instance with an iterable type, else False.
        """
        if isinstance(self.assigned_value, CanBehaveLikeAVariable):
            return self.assigned_value._is_iterable_
        elif not isinstance(self.assigned_value, Match) and is_iterable(
            self.assigned_value
        ):
            return True
        elif (
            isinstance(self.assigned_value, Match)
            and self.assigned_value._variable_._is_iterable_
        ):
            return True
        return False

    @cached_property
    def is_type_filter_needed(self):
        """
        :return: True if a type filter condition is needed for the attribute assignment, else False.
        """
        attr_type = self.attr._type_
        return (not attr_type) or (
            (self.assigned_value._type_ and self.assigned_value._type_ is not attr_type)
            and issubclass(self.assigned_value._type_, attr_type)
        )


@dataclass
class Select(Match[T], Selectable[T]):
    """
    This is a Match with the addition that the matched entity is selected in the result.
    """

    _is_selected_: bool = field(init=False, default=True)

    def __post_init__(self):
        """
        This is needed to prevent the SymbolicExpression __post_init__ from being called which will make a node out of
        this instance, and that is not what we want.
        """
        ...

    def _resolve_(
        self,
        variable: Optional[CanBehaveLikeAVariable] = None,
        parent: Optional[Match] = None,
    ):
        super()._resolve_(variable, parent)
        variable = variable or self._variable_
        if not self._var_:
            self._var_ = variable

    def _evaluate__(
        self,
        sources: Optional[Dict[int, Any]] = None,
        parent: Optional[SymbolicExpression] = None,
    ) -> Iterable[OperationResult]:
        yield from self._variable_._evaluate__(sources, parent)

    @property
    def _name_(self) -> str:
        return self._var_._name_

    @cached_property
    def _all_variable_instances_(self) -> List[CanBehaveLikeAVariable[T]]:
        return self._var_._all_variable_instances_


def matching(
    type_: Union[Type[T], CanBehaveLikeAVariable[T], Any, None] = None,
) -> Union[Type[T], CanBehaveLikeAVariable[T], Match[T]]:
    """
    Create and return a Match instance that looks for the pattern provided by the type and the
    keyword arguments.

    :param type_: The type of the variable (i.e., The class you want to instantiate).
    :return: The Match instance.
    """
    return entity_matching(type_, None)


def match_any(
    type_: Union[Type[T], CanBehaveLikeAVariable[T], Any, None] = None,
) -> Union[Type[T], CanBehaveLikeAVariable[T], Match[T]]:
    """
    Equivalent to matching(type_) but for existential checks.
    """
    match_ = matching(type_)
    match_._existential_ = True
    return match_


def match_all(
    type_: Union[Type[T], CanBehaveLikeAVariable[T], Any, None] = None,
) -> Union[Type[T], CanBehaveLikeAVariable[T], Match[T]]:
    """
    Equivalent to matching(type_) but for universal checks.
    """
    match_ = matching(type_)
    match_._universal_ = True
    return match_


def select(
    *variables: Union[T, CanBehaveLikeAVariable[T], Match[T]],
) -> Match[T]:
    """
    Equivalent to matching(type_) and selecting the variable to be included in the result.
    """
    for variable in variables:
        variable._set_as_selected_()
    return variables[0]._root_match_


def select_any(
    type_: Union[Type[T], CanBehaveLikeAVariable[T], Any, None] = None,
) -> Union[Type[T], CanBehaveLikeAVariable[T], Select[T]]:
    """
    Equivalent to select(type_) but for existential checks.
    """
    select_ = select(type_)
    select_._existential_ = True
    return select_


def select_all(
    type_: Union[Type[T], CanBehaveLikeAVariable[T], Any, None] = None,
) -> Union[Type[T], CanBehaveLikeAVariable[T], Select[T]]:
    """
    Equivalent to select(type_) but for universal checks.
    """
    select_ = select(type_)
    select_._universal_ = True
    return select_


def entity_matching(
    type_: Union[Type[T], CanBehaveLikeAVariable[T]], domain: DomainType
) -> Union[Type[T], CanBehaveLikeAVariable[T], Match[T]]:
    """
    Same as :py:func:`krrood.entity_query_language.match.match` but with a domain to use for the variable created
     by the match.

    :param type_: The type of the variable (i.e., The class you want to instantiate).
    :param domain: The domain used for the variable created by the match.
    :return: The MatchEntity instance.
    """
    if isinstance(type_, CanBehaveLikeAVariable):
        return Match(type_._type_, _domain_=domain, _variable_=type_)
    elif type_ and not isinstance(type_, type):
        return Match(type_, _domain_=domain, _variable_=Literal(type_))
    return Match(type_, _domain_=domain)


def entity_selection(
    type_: Union[Type[T], CanBehaveLikeAVariable[T]], domain: DomainType
) -> Union[Type[T], CanBehaveLikeAVariable[T], Select[T]]:
    """
    Same as :py:func:`krrood.entity_query_language.match.entity_matching` but also selecting the variable to be
     included in the result.
    """
    if isinstance(type_, CanBehaveLikeAVariable):
        return Select(type_._type_, _domain_=domain, _variable_=type_)
    elif type_ and not isinstance(type_, type):
        return Select(type_, _domain_=domain, _variable_=Literal(type_))
    return Select(type_, _domain_=domain)


EntityType = Union[SetOf[T], Entity[T], T, Iterable[T], Type[T], Match[T]]
"""
The possible types for entities.
"""
