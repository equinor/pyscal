"""Container class for list of Pyscal objects"""

from __future__ import division, absolute_import
from __future__ import print_function

import os

import logging

import six
import pandas as pd

from pyscal import WaterOil, GasOil, GasWater, WaterOilGas, SCALrecommendation

PYSCAL_OBJECTS = [WaterOil, GasOil, GasWater, WaterOilGas, SCALrecommendation]

logging.basicConfig()
logger = logging.getLogger(__name__)


class PyscalList(object):
    """Container class for a list of WaterOilGas objects.

    Essentially this is a list of objects of equal type, and all
    being pyscal objects WaterOil, GasOil, WaterOilGas or SCALrecommendation

    It is possible to ask this list class for SWOF++ printouts,
    and it will call SWOF on each element succesively.

    Args:
        pyscal_list (list): List of objects if already ready. Can be empty or None.
    """

    def __init__(self, pyscal_list=None):
        self.pyscaltype = None
        if isinstance(pyscal_list, list):
            for pyscal_obj in pyscal_list:
                self.append(pyscal_obj)
        else:
            self.pyscal_list = []

    def append(self, pyscal_obj):
        """Append a pyscal object to the list

        Args:
            pyscal_obj (WaterOil, GasOil, WaterOilGas or SCALrecommendation)

        Raises:
            ValueError if the type of the incoming object does not
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
            logger.error(
                "Trying to add %s to list of %s objects",
                type(pyscal_obj),
                self.pyscaltype,
            )
            raise ValueError
        self.pyscal_list.append(pyscal_obj)

    def df(self):
        """Dump dataframes of generated relperm data

        Column names are compatible with ecl2df.satfunc. Always uppercase
        and capillary pressure is PCOW or PCOG (wateroil vs gasoil)

        If the PyscalList contains SCALrecommendations, the CASE column
        will contain the strings 'pess', 'base' and 'opt' (independent of
        any alias name potentially used in an input xlsx/csv)

        Returns:
            pd.DataFrame
        """
        # Names of dataframe columns in wateroil/gasoil.table:
        wateroil_pyscal_cols = {"sw", "krw", "krow", "pc"}
        gasoil_pyscal_cols = {"sg", "krg", "krog", "pc"}

        # Renamers applied to the returned dataframe:
        gasoil_col_renamer = {"sg": "SG", "krg": "KRG", "krog": "KROG", "pc": "PCOG"}
        wateroil_col_renamer = {"sw": "SW", "krw": "KRW", "krow": "KROW", "pc": "PCOW"}

        # Sort order for rows in returned dataframe:
        sort_candidates = ["SATNUM", "CASE", "KEYWORD", "SW", "SG", "SL"]

        df_list = []
        if self.pyscaltype == WaterOilGas:
            for (satnum, wateroilgas) in enumerate(self.pyscal_list):
                wateroil_cols = set(wateroilgas.wateroil.table.columns).intersection(
                    wateroil_pyscal_cols
                )
                gasoil_cols = set(wateroilgas.gasoil.table.columns).intersection(
                    gasoil_pyscal_cols
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
                gasoil_cols = set(scalrec.base.gasoil.table.columns).intersection(
                    gasoil_pyscal_cols
                )
                wateroil_cols = set(scalrec.base.wateroil.table.columns).intersection(
                    wateroil_pyscal_cols
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
                wateroil_cols = set(wateroil.table.columns).intersection(
                    wateroil_pyscal_cols
                )
                df_list.append(
                    wateroil.table[wateroil_cols]
                    .assign(SATNUM=satnum + 1)
                    .rename(wateroil_col_renamer, axis="columns")
                )
        elif self.pyscaltype == GasOil:
            for (satnum, gasoil) in enumerate(self.pyscal_list):
                gasoil_cols = set(gasoil.table.columns).intersection(gasoil_pyscal_cols)
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

    def dump_family_1(self, filename=None, slgof=False):
        """Dumps family 1 Eclipse saturation tables to one
        filename. This means SWOF + SGOF (SGOF only if relevant)

        Args:
            filename (str): Filename for the output to be given to Eclips 100
            slgof (bool): Set to true of SLGOF is wanted instead of SGOF
        """
        if self.pyscaltype == SCALrecommendation:
            logger.error(
                "You need to interpolate before you can dump a SCAL recommendation"
            )
            raise TypeError
        if self.pyscaltype == WaterOilGas:
            # WaterOilGas can be of type WaterOil when it emerges
            # from a SCAL recommendation, do a fragile test:
            if self.pyscal_list[0].gasoil is None:
                family_1_str = self.SWOF()
                keywords = "SWOF"
            elif self.pyscal_list[0].wateroil is None:
                family_1_str = self.SGOF()
                keywords = "SGOF"
            elif not slgof:
                family_1_str = self.SWOF() + "\n" + self.SGOF()
                keywords = "SWOF and SGOF"
            else:
                family_1_str = self.SWOF() + "\n" + self.SLGOF()
                keywords = "SWOF and SLGOF"
        if self.pyscaltype == WaterOil:
            family_1_str = self.SWOF()
            keywords = "SWOF"
        if self.pyscaltype == GasOil:
            family_1_str = self.SGOF()
            keywords = "SGOF"
            if slgof:
                logger.warning("SLGOF not meaningful for GasOil. Ignored")
        if self.pyscaltype == GasWater:
            msg = "Family 1 output not possible for GasWater"
            logger.error(msg)
            raise ValueError(msg)
        if filename is not None:
            logger.info(
                "Dumping family 1 keywords (%s) for %d SATNUMs to %s",
                keywords,
                len(self),
                filename,
            )
            directory = os.path.dirname(filename)
            if directory and not os.path.exists(directory):
                os.makedirs(directory)
            with open(filename, "w") as file_h:
                file_h.write(six.ensure_str(family_1_str))
        return family_1_str

    def dump_family_2(self, filename=None):
        """Dumps family 2 Eclipse saturation tables to one
        filename. This means SWFN + SGFN + SOF3 (SOF3 only for WaterOilGas)

        Relevant for WaterOilGas and GasWater.

        Args:
            filename (str): Filename for the output to be given to Eclipse 100
        """
        if self.pyscaltype == SCALrecommendation:
            logger.error(
                "You need to interpolate before you can dump a SCAL recommendation"
            )
            raise TypeError
        if self.pyscaltype == WaterOilGas:
            family_2_str = self.SWFN() + "\n" + self.SGFN() + "\n" + self.SOF3()
            keywords = "SWFN, SGFN and SOF3"
        elif self.pyscaltype == GasWater:
            family_2_str = self.SWFN() + "\n" + self.SGFN() + "\n"
            keywords = "SWFN and SGFN"
        else:
            logger.error("Family 2 only supported for WaterOilGas and GasWater")
            raise ValueError
        if filename is not None:
            logger.info(
                "Dumping family 2 keywords (%s) for %d SATNUMs to %s",
                keywords,
                len(self),
                filename,
            )
            directory = os.path.dirname(filename)
            if directory and not os.path.exists(directory):
                os.makedirs(directory)
            with open(filename, "w") as file_h:
                file_h.write(six.ensure_str(family_2_str))
        return family_2_str

    def interpolate(self, int_params_wo, int_params_go=None, h=None):
        """This function will interpolate each SCALrecommendation
        object to the chosen parameters

        This only works on lists of SCALrecommendation objects

        Args:
            int_params_wo (float or list of float): Interpolation parameters
                for wateroil, or for both. If list,
                separate parameter for each SATNUM. All numbers between
                -1 and 1.
            int_params_go (float or list of float): If specified, will
                be used for GasOil interpolation.
            h (float): Saturation step-length

        Returns:
            PyscalList of type WaterOilGas, with the same length.
        """

        if self.pyscaltype != SCALrecommendation:
            logger.error("Can only interpolate PyscalList of type SCALrecommendation")
            raise TypeError
        if not isinstance(int_params_wo, list):
            int_params_wo = [int_params_wo] * self.__len__()
        if isinstance(int_params_wo, list) and len(int_params_wo) == 1:
            int_params_wo = int_params_wo * self.__len__()
        if not isinstance(int_params_go, list):
            int_params_go = [int_params_go] * len(self)
        if isinstance(int_params_go, list) and len(int_params_go) == 1:
            int_params_go = int_params_go * self.__len__()
        if 1 < len(int_params_wo) < len(self):
            logger.error(
                "Too few interpolation parameters given for WaterOil %s",
                str(int_params_wo),
            )
            raise ValueError
        if len(int_params_wo) > len(self):
            logger.error(
                "Too many interpolation parameters given for WaterOil %s",
                str(int_params_wo),
            )
            raise ValueError
        if 1 < len(int_params_go) < len(self):
            logger.error(
                "Too few interpolation parameters given for GasOil %s",
                str(int_params_go),
            )
            raise ValueError
        if len(int_params_go) > len(self):
            logger.error(
                "Too many interpolation parameters given for GasOil %s",
                str(int_params_go),
            )
            raise ValueError
        wog_list = PyscalList()
        for (satnum, scalrec) in enumerate(self.pyscal_list):
            wog_list.append(
                scalrec.interpolate(int_params_wo[satnum], int_params_go[satnum], h=h)
            )
        return wog_list

    def make_ecl_output(self, keyword, write_to_filename=None, gaswater=False):
        """Internal helper function for constructing strings and writing to disk"""
        if self.pyscaltype == SCALrecommendation:
            logger.error(
                "You need to interpolate before you can dump a SCAL recommendation"
            )
            raise TypeError
        first_obj = self.pyscal_list[0]
        outputter = getattr(first_obj, keyword)
        if gaswater:
            string = outputter(header=True, gaswater=gaswater)
        else:
            string = outputter(header=True)
        if len(self.pyscal_list) > 1:
            for pyscal_obj in self.pyscal_list[1:]:
                outputter = getattr(pyscal_obj, keyword)
                if gaswater:
                    string += outputter(header=False, gaswater=gaswater)
                else:
                    string += outputter(header=False)
        if write_to_filename:
            directory = os.path.dirname(write_to_filename)
            if directory and not os.path.exists(directory):
                os.makedirs(directory)
            with open(write_to_filename, "w") as file_h:
                file_h.write(six.ensure_str(string))
        return string

    def SWOF(self, write_to_filename=None):
        """Make SWOF string and optionally print to file"""
        return self.make_ecl_output("SWOF", write_to_filename)

    def SGOF(self, write_to_filename=None):
        """Make SGOF string and optionally print to file"""
        return self.make_ecl_output("SGOF", write_to_filename)

    def SLGOF(self, write_to_filename=None):
        """Make SLGOF string and optionally print to file"""
        return self.make_ecl_output("SLGOF", write_to_filename)

    def SGFN(self, write_to_filename=None, gaswater=False):
        """Make SGFN string and optionally print to file"""
        return self.make_ecl_output("SGFN", write_to_filename, gaswater=gaswater)

    def SWFN(self, write_to_filename=None, gaswater=False):
        """Make SWFN string and optionally print to file"""
        return self.make_ecl_output("SWFN", write_to_filename, gaswater=gaswater)

    def SOF3(self, write_to_filename=None):
        """Make SOF3 string and optionally print to file"""
        return self.make_ecl_output("SOF3", write_to_filename)

    def __len__(self):
        """Return the count of Pyscal objects in the list"""
        return len(self.pyscal_list)

    def __getitem__(self, satnum_idx):
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
