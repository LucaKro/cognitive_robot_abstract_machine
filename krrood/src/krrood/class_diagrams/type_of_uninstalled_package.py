"""
Stand-ins for types that a module imports from a package that is not installed.
"""

from __future__ import annotations

import ast
import importlib.util
from dataclasses import dataclass
from functools import cache

from krrood.utils import module_level_imports

# %% the import of a type from an uninstalled package


@dataclass(frozen=True)
class ImportFromUninstalledPackage:
    """
    A module-level import of a name from a package that is not installed, such as an
    optional dependency imported only for type checking.
    """

    module_name: str
    """
    The absolute name of the module the name is imported from.
    """

    name: str
    """
    The name as that module defines it.
    """

    @classmethod
    def binding(
        cls, bound_name: str, source_path: str
    ) -> ImportFromUninstalledPackage | None:
        """
        :param bound_name: The name the import binds in the module, directly or as an
            alias.
        :param source_path: The source file of the module.
        :return: The import binding the name, or ``None`` if no absolute module-level
            import binds it or its package is installed.
        """
        with open(source_path) as source_file:
            tree = ast.parse(source_file.read(), filename=source_path)
        imports_binding_the_name = (
            cls(statement.module, alias.name)
            for statement in module_level_imports(tree)
            if isinstance(statement, ast.ImportFrom) and not statement.level
            for alias in statement.names
            if (alias.asname or alias.name) == bound_name
        )
        return next(
            (
                found
                for found in imports_binding_the_name
                if importlib.util.find_spec(found.package_name) is None
            ),
            None,
        )

    @property
    def package_name(self) -> str:
        """
        The top-level package the name is imported from.
        """
        return self.module_name.partition(".")[0]


# %% the stand-in type


class TypeOfUninstalledPackage:
    """
    The base of every stand-in for a type imported from a package that is not installed.

    A stand-in has the module and name of the type it stands in for, and is never
    instantiated.
    """

    @classmethod
    @cache
    def standing_in_for(
        cls, uninstalled_import: ImportFromUninstalledPackage
    ) -> type[TypeOfUninstalledPackage]:
        """
        :param uninstalled_import: The import of the type.
        :return: The one stand-in for the imported type.
        """
        return type(
            uninstalled_import.name,
            (cls,),
            {
                "__module__": uninstalled_import.module_name,
                "__qualname__": uninstalled_import.name,
            },
        )
