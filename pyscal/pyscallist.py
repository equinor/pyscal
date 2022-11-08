"""Container class for list of Pyscal objects"""

import warnings
from pathlib import Path
from typing import List, Optional, Type, Union

import pandas as pd

from pyscal import (
    GasOil,
    GasWater,
    SCALrecommendation,
    WaterOil,
    WaterOilGas,
    getLogger_pyscal,
)

PYSCAL_OBJECTS = [WaterOil, GasOil, GasWater, WaterOilGas, SCALrecommendation]

PyscalObjects = Union[WaterOil, GasOil, GasWater, WaterOilGas, SCALrecommendation]

logger = getLogger_pyscal(__name__)

warnings.filterwarnings("default", category=DeprecationWarning, module="pyscal")


class PyscalList(object):
    """Container class for a list of WaterOilGas objects.

    Essentially this is a list of objects of equal type, and all
    being pyscal objects WaterOil, GasOil, WaterOilGas or SCALrecommendation

    It is possible to ask this list class for SWOF++ printouts,
    and it will call SWOF on each element succesively.

    Args:
        pyscal_list (list): List of objects if already ready. Can be empty or None.
    """

    def __init__(self, pyscal_list: Optional[List[PyscalObjects]] = None):
        self.pyscaltype: Optional[Type] = None
        self.pyscal_list: List[PyscalObjects] = []
        if isinstance(pyscal_list, list):
            for pyscal_obj in pyscal_list:
                self.append(pyscal_obj)
        if isinstance(pyscal_list, PyscalList):
            for idx in range(len(pyscal_list)):
                self.append(pyscal_list[idx + 1])

    def append(self, pyscal_obj: Optional[PyscalObjects]) -> None:
        """Append a pyscal object to the list

        Args:
            pyscal_obj

        Raises:
            ValueError
                If the type of the incoming object does not
                match existing objects in the list
        """
        if pyscal_obj is None:
            return
        if isinstance(pyscal_obj, list):
            # Recursion
            for pyscal_obj_sub in pyscal_obj:
                self.append(pyscal_obj_sub)
            return
        if not isinstance(pyscal_obj, tuple(PYSCAL_OBJECTS)):
            raise ValueError("Not a pyscal object: " + str(pyscal_obj))
        if not self.pyscaltype:
            self.pyscaltype = type(pyscal_obj)
            # Beware, this list can be of type WaterOilGas, with
            # WaterOilGas objects where gasoil is None, effectively
            # making that object a WaterOil object.
        if not isinstance(pyscal_obj, self.pyscaltype):
            raise ValueError(
                f"Trying to add {type(pyscal_obj)} to list "
                f"of {self.pyscaltype} objects."
            )
        self.pyscal_list.append(pyscal_obj)

    def df(self) -> pd.DataFrame:
        """Dump dataframes of generated relperm data

        Column names are compatible with ecl2df.satfunc. Always uppercase
        and capillary pressure is PCOW or PCOG (wateroil vs gasoil)

        If the PyscalList contains SCALrecommendations, the CASE column
        will contain the strings 'pess', 'base' and 'opt' (independent of
        any alias name potentially used in an input xlsx/csv)
        """
        # Names of dataframe columns in wateroil/gasoil.table:
        wateroil_pyscal_cols = {"SW", "KRW", "KROW", "PC"}
        gasoil_pyscal_cols = {"SG", "KRG", "KROG", "PC"}

        # Renamers applied to the returned dataframe:
        gasoil_col_renamer = {"SG": "SG", "KRG": "KRG", "KROG": "KROG", "PC": "PCOG"}
        wateroil_col_renamer = {"SW": "SW", "KRW": "KRW", "KROW": "KROW", "PC": "PCOW"}

        # Sort order for rows in returned dataframe:
        sort_candidates = ["SATNUM", "CASE", "KEYWORD", "SW", "SG", "SL"]

        df_list = []
        if self.pyscaltype == WaterOilGas:
            for (satnum, wateroilgas) in enumerate(self.pyscal_list):
                assert isinstance(wateroilgas, WaterOilGas)
                assert wateroilgas.wateroil is not None
                assert wateroilgas.gasoil is not None
                wateroil_cols = list(
                    set(wateroilgas.wateroil.table.columns).intersection(
                        wateroil_pyscal_cols
                    )
                )
                gasoil_cols = list(
                    set(wateroilgas.gasoil.table.columns).intersection(
                        gasoil_pyscal_cols
                    )
                )
                df_list.append(
                    wateroilgas.gasoil.table[gasoil_cols]
                    .assign(SATNUM=satnum + 1)
                    .rename(gasoil_col_renamer, axis="columns")
                )
                df_list.append(
                    wateroilgas.wateroil.table[wateroil_cols]
                    .assign(SATNUM=satnum + 1)
                    .rename(wateroil_col_renamer, axis="columns")
                )
        elif self.pyscaltype == SCALrecommendation:
            for (satnum, scalrec) in enumerate(self.pyscal_list):
                assert isinstance(scalrec, SCALrecommendation)
                assert scalrec.low is not None
                assert scalrec.base is not None
                assert scalrec.high is not None
                assert scalrec.low.wateroil is not None
                assert scalrec.low.gasoil is not None
                assert scalrec.base.wateroil is not None
                assert scalrec.base.gasoil is not None
                assert scalrec.high.wateroil is not None
                assert scalrec.high.gasoil is not None
                gasoil_cols = list(
                    set(scalrec.base.gasoil.table.columns).intersection(
                        gasoil_pyscal_cols
                    )
                )
                wateroil_cols = list(
                    set(scalrec.base.wateroil.table.columns).intersection(
                        wateroil_pyscal_cols
                    )
                )
                df_list.append(
                    scalrec.low.gasoil.table[gasoil_cols]
                    .assign(SATNUM=satnum + 1, CASE="pess")
                    .rename(gasoil_col_renamer, axis="columns")
                )
                df_list.append(
                    scalrec.base.gasoil.table[gasoil_cols]
                    .assign(SATNUM=satnum + 1, CASE="base")
                    .rename(gasoil_col_renamer, axis="columns")
                )
                df_list.append(
                    scalrec.high.gasoil.table[gasoil_cols]
                    .assign(SATNUM=satnum + 1, CASE="opt")
                    .rename(gasoil_col_renamer, axis="columns")
                )

                df_list.append(
                    scalrec.low.wateroil.table[wateroil_cols]
                    .assign(SATNUM=satnum + 1, CASE="pess")
                    .rename(wateroil_col_renamer, axis="columns")
                )
                df_list.append(
                    scalrec.base.wateroil.table[wateroil_cols]
                    .assign(SATNUM=satnum + 1, CASE="base")
                    .rename(wateroil_col_renamer, axis="columns")
                )
                df_list.append(
                    scalrec.high.wateroil.table[wateroil_cols]
                    .assign(SATNUM=satnum + 1, CASE="opt")
                    .rename(wateroil_col_renamer, axis="columns")
                )
        elif self.pyscaltype == WaterOil:
            for (satnum, wateroil) in enumerate(self.pyscal_list):
                assert isinstance(wateroil, WaterOil)
                assert wateroil is not None
                wateroil_cols = list(
                    set(wateroil.table.columns).intersection(wateroil_pyscal_cols)
                )
                df_list.append(
                    wateroil.table[wateroil_cols]
                    .assign(SATNUM=satnum + 1)
                    .rename(wateroil_col_renamer, axis="columns")
                )
        elif self.pyscaltype == GasOil:
            for (satnum, gasoil) in enumerate(self.pyscal_list):
                assert isinstance(gasoil, GasOil)
                assert gasoil is not None
                gasoil_cols = list(
                    set(gasoil.table.columns).intersection(gasoil_pyscal_cols)
                )
                df_list.append(
                    gasoil.table[gasoil_cols]
                    .assign(SATNUM=satnum + 1)
                    .rename(gasoil_col_renamer, axis="columns")
                )
        dframe = pd.concat(df_list, sort=False, ignore_index=True)
        sort_rows_on = [colname for colname in sort_candidates if colname in dframe]
        if sort_rows_on:
            dframe.sort_values(sort_rows_on, inplace=True)
        return dframe

    def relevant_keywords(self, family: int = 1, slgof: bool = False) -> List[str]:
        """Construct a list of relevant Eclipse keywords for the data in this
        Pyscallist object. This depends on the Pyscaltype, and which family is
        requested"""

        if family not in [1, 2]:
            raise ValueError("Family must be either 1 or 2")

        if self.pyscaltype == WaterOilGas:
            # WaterOilGas can be of type WaterOil or GasOil when it emerges
            # from a SCAL recommendation, signified by None-ness of attributes
            if family == 2:
                return ["SWFN", "SGFN", "SOF3"]
            if self.pyscal_list[0].gasoil is None:  # type: ignore
                return ["SWOF"]
            if self.pyscal_list[0].wateroil is None:  # type: ignore
                return ["SGOF"]
            if not slgof:
                return ["SWOF", "SGOF"]
            return ["SWOF", "SLGOF"]

        if self.pyscaltype == WaterOil:
            if family == 2:
                raise ValueError("Family 2 only supported for WaterOilGas and GasWater")
            return ["SWOF"]

        if self.pyscaltype == GasOil:
            if family == 2:
                raise ValueError("Family 2 only supported for WaterOilGas and GasWater")
            if slgof:
                raise ValueError("SLGOF not meaningful for GasOil")
            return ["SGOF"]

        assert self.pyscaltype == GasWater
        if family == 2:
            return ["SWFN", "SGFN"]
        raise ValueError("Family 1 output not possible for GasWater")

    def build_eclipse_data(self, family: int = 1, slgof: bool = False) -> str:
        """Construct Eclipse keywords and data for relative permeability
        properties of family 1 or 2 type.

        Args:
            slgof: Set to true of SLGOF is wanted instead of SGOF. Only applicable
            if family is 1.
        """
        if family not in [1, 2]:
            raise ValueError("Family must be either 1 or 2")
        if len(self) == 0:
            return ""
        if self.pyscaltype == SCALrecommendation:
            raise TypeError(
                "You need to interpolate before you can dump a SCAL recommendation"
            )
        if family == 2 and slgof is True:
            raise ValueError("SLGOF not meaningful for family 2")
        keywords = self.relevant_keywords(family=family, slgof=slgof)
        logger.info(
            "Keywords %s (family %d) for %d SATNUMs generated",
            ", ".join(keywords),
            family,
            len(self),
        )
        return "\n".join([getattr(self, keyword)() for keyword in keywords])

    def dump_family_1(self, filename: Optional[str] = None, slgof: bool = False) -> str:
        """Dumps family 1 Eclipse saturation tables to one
        filename. This means SWOF + SGOF (SGOF only if relevant)

        This function is deprecated. Use build_eclipse_data() and write to
        disk in calling code.

        Args:
            filename: Filename for the output to be given to Eclipse 100
            slgof: Set to true of SLGOF is wanted instead of SGOF
        """
        warnings.warn("dump_family_1() is deprecated", DeprecationWarning)
        string = self.build_eclipse_data(family=1, slgof=slgof)
        if filename is not None:
            if not Path(filename).parent.exists():
                warnings.warn(
                    "Please create the output directory prior to calling pyscal.",
                    DeprecationWarning,
                )
                Path(filename).parent.mkdir(exist_ok=True, parents=True)
            Path(filename).write_text(string, encoding="utf-8")
        return string

    def dump_family_2(self, filename: Optional[str] = None) -> str:
        """Dumps family 2 Eclipse saturation tables to one
        filename. This means SWFN + SGFN + SOF3 (SOF3 only for WaterOilGas)

        Relevant for WaterOilGas and GasWater.

        Args:
            filename (str): Filename for the output to be given to Eclipse 100
        """
        warnings.warn("dump_family_2() is deprecated", DeprecationWarning)
        string = self.build_eclipse_data(family=2, slgof=False)
        if filename is not None:
            if not Path(filename).parent.exists():
                warnings.warn(
                    "Please create the output directory prior to calling pyscal.",
                    DeprecationWarning,
                )
                Path(filename).parent.mkdir(exist_ok=True, parents=True)
            Path(filename).write_text(string, encoding="utf-8")
        return string

    def interpolate(
        self,
        int_params_wo: Union[float, int, List[float]],
        int_params_go: Optional[Union[float, int, List[Optional[float]]]] = None,
        h: Optional[float] = None,
    ) -> "PyscalList":
        """This function will interpolate each SCALrecommendation
        object to the chosen parameters

        This only works on lists of SCALrecommendation objects

        Args:
            int_params_wo: Interpolation parameters for wateroil, or for
                both. If list, separate parameter for each SATNUM. All
                numbers between -1 and 1 (inclusive).
            int_params_go: If specified, will be used for GasOil interpolation.
            h: Saturation step-length

        Returns:
            PyscalList of type WaterOilGas, with the same length.
        """

        if self.pyscaltype != SCALrecommendation:
            raise TypeError(
                "Can only interpolate PyscalList of type SCALrecommendation"
            )
        if isinstance(int_params_wo, (float, int)):
            int_params_wo = [int_params_wo] * self.__len__()
        if isinstance(int_params_wo, list) and len(int_params_wo) == 1:
            int_params_wo = int_params_wo * self.__len__()
        if int_params_go is None or isinstance(int_params_go, (float, int)):
            int_params_go = [int_params_go] * len(self)
        if isinstance(int_params_go, list) and len(int_params_go) == 1:
            int_params_go = int_params_go * self.__len__()
        if 1 < len(int_params_wo) < len(self):
            raise ValueError(
                f"Too few interpolation parameters given for WaterOil {int_params_wo}"
            )
        if len(int_params_wo) > len(self):
            raise ValueError(
                f"Too many interpolation parameters given for WaterOil {int_params_wo}",
            )
        if 1 < len(int_params_go) < len(self):
            raise ValueError(
                f"Too few interpolation parameters given for GasOil {int_params_go}"
            )
        if len(int_params_go) > len(self):
            raise ValueError(
                f"Too many interpolation parameters given for GasOil {int_params_go}"
            )
        wog_list: PyscalList = PyscalList()
        for (satnum, scalrec) in enumerate(self.pyscal_list):
            assert isinstance(scalrec, SCALrecommendation)
            wog_list.append(
                scalrec.interpolate(int_params_wo[satnum], int_params_go[satnum], h=h)
            )
        return wog_list

    def _make_ecl_output(
        self,
        keyword: str,
        write_to_filename: Optional[str] = None,  # Deprecated
    ) -> str:
        """Internal helper function for constructing Eclipse include file strings
        for individual keywords.

        build_eclipse_data() will use this function.
        """
        if self.pyscaltype == SCALrecommendation:
            raise TypeError(
                "You need to interpolate before you can dump a SCAL recommendation"
            )
        first_obj = self.pyscal_list[0]
        outputter = getattr(first_obj, keyword)
        string = outputter(header=True)
        if len(self.pyscal_list) > 1:
            for pyscal_obj in self.pyscal_list[1:]:
                outputter = getattr(pyscal_obj, keyword)
                string += outputter(header=False)
        if write_to_filename:
            warnings.warn(
                "Writing to files in pyscallist is deprecated", DeprecationWarning
            )
            Path(write_to_filename).parent.mkdir(parents=True, exist_ok=True)
            Path(write_to_filename).write_text(string, encoding="utf-8")
        return string

    def SWOF(self, write_to_filename: Optional[str] = None) -> str:
        """Build SWOF string"""
        # _make_ecl_output() will warn about non-None filename being deprecated
        return self._make_ecl_output("SWOF", write_to_filename)

    def SGOF(self, write_to_filename: Optional[str] = None) -> str:
        """Build SGOF string"""
        return self._make_ecl_output("SGOF", write_to_filename)

    def SLGOF(self, write_to_filename: Optional[str] = None) -> str:
        """Build SLGOF string"""
        return self._make_ecl_output("SLGOF", write_to_filename)

    def SGFN(self, write_to_filename: Optional[str] = None) -> str:
        """Build SGFN string"""
        return self._make_ecl_output("SGFN", write_to_filename)

    def SWFN(self, write_to_filename: Optional[str] = None) -> str:
        """Build SWFN string"""
        return self._make_ecl_output("SWFN", write_to_filename)

    def SOF3(self, write_to_filename: Optional[str] = None) -> str:
        """Build SOF3 string"""
        return self._make_ecl_output("SOF3", write_to_filename)

    def __len__(self) -> int:
        """Return the count of Pyscal objects in the list"""
        return len(self.pyscal_list)

    def __getitem__(self, satnum_idx) -> PyscalObjects:
        """Get a specific List member.

        The indexing starts at 1, not zero, similar
        to how SATNUMs are indexed.

        Args:
            satnum_idx (int): Index for wanted SATNUM. Starts at 1

        Returns:
            WaterOilGas, GasOil, WaterOil or SCALrecommendation, depending
                on self.pyscaltype.
        """
        if satnum_idx < 1:
            e_msg = "SATNUM must be 1 or higher"
            logger.error(e_msg)
            raise IndexError(e_msg)
        if satnum_idx > self.__len__():
            e_msg = "SATNUM index out of range, length is " + str(self.__len__())
            logger.error(e_msg)
            raise IndexError(e_msg)
        return self.pyscal_list[satnum_idx - 1]
