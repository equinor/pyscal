"""Container class for list of Pyscal objects"""

from __future__ import division, absolute_import
from __future__ import print_function

import os

import logging

import six

from pyscal import WaterOilGas, WaterOil, GasOil, SCALrecommendation

PYSCAL_OBJECTS = [WaterOil, GasOil, WaterOilGas, SCALrecommendation]

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
        if not isinstance(pyscal_obj, self.pyscaltype):
            logger.error(
                "Trying to add %s to list of %s objects",
                type(pyscal_obj),
                self.pyscaltype,
            )
            raise ValueError
        self.pyscal_list.append(pyscal_obj)

    def dump_family_1(self, filename=None, slgof=False):
        """Dumps family 1 Eclipse saturation tables to one
        filename. This means SWOF + SGOF (SGOF only if relevant)

        Args:
            filename (str): Filename for the output to be given to Eclips 100
            slgof (bool): Set to true of SLGOF is wanted instead of SGOF
            interpolate (float, tuple of two floats, or list of floats or
                list of tuples):  Interpolation parameter(s), tuple refers
                to different parameter for wateroil and gasoil, list refers to
                indidual SATNUMs
        """
        logger.info("FOOBAR")
        if self.pyscaltype == SCALrecommendation:
            logger.error(
                "You need to interpolate before you can dump a SCAL recommendation"
            )
            raise TypeError
        if self.pyscaltype == WaterOilGas:
            if not slgof:
                family_1_str = self.SWOF() + "\n" + self.SGOF()
            else:
                family_1_str = self.SWOF() + "\n" + self.SLGOF()
        if self.pyscaltype == WaterOil:
            family_1_str = self.SWOF()
        if self.pyscaltype == GasOil:
            family_1_str = self.SGOF
            if slgof:
                logger.warning("SLGOF not meaningful for GasOil. Ignored")
        if filename is not None:
            logger.info(
                "Dumping family 1 keywords for %d SATNUMs to %s", len(self), filename
            )
            directory = os.path.dirname(filename)
            if directory and not os.path.exists(directory):
                os.makedirs(directory)
            with open(filename, "w") as file_h:
                file_h.write(six.ensure_str(family_1_str))
        return family_1_str

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
        if int_params_go is None:
            int_params_go = int_params_wo
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
            # Revise (remove?) these 4 lines while solving github issue #105
            if wog_list.pyscal_list[satnum].wateroil.tag == "":
                wog_list.pyscal_list[satnum].wateroil.tag = "SATNUM " + str(satnum + 1)
            if wog_list.pyscal_list[satnum].gasoil.tag == "":
                wog_list.pyscal_list[satnum].gasoil.tag = "SATNUM " + str(satnum + 1)
        return wog_list

    def dump_family_2(self, filename=None):
        """Dumps family 2 Eclipse saturation tables to one
        filename. This means SWFN + SGFN + SOF3

        Can interpolate if is a list of SCAL recommendations.

        Args:
            filename (str): Filename for the output to be given to Eclips 100
            interpolate (float, tuple of two floats, or list of floats or
                list of tuples):  Interpolation parameter(s), tuple refers
                to different parameter for wateroil and gasoil, list refers to
                indidual SATNUMs
        """
        if self.pyscaltype == SCALrecommendation:
            logger.error(
                "You need to interpolate before you can dump a SCAL recommendation"
            )
            raise TypeError
        if self.pyscaltype != WaterOilGas:
            logger.error("Family 2 only supported for WaterOilGas")
            raise ValueError
        family_2_str = self.SWFN() + "\n" + self.SGFN() + "\n" + self.SOF3()
        if filename is not None:
            directory = os.path.dirname(filename)
            if directory and not os.path.exists(directory):
                os.makedirs(directory)
            with open(filename, "w") as file_h:
                file_h.write(six.ensure_str(family_2_str))
        return family_2_str

    def make_ecl_output(self, keyword, write_to_filename=None):
        """Internal helper function for constructing strings and writing to disk"""
        if self.pyscaltype == SCALrecommendation:
            logger.error(
                "You need to interpolate before you can dump a SCAL recommendation"
            )
            raise TypeError
        first_obj = self.pyscal_list[0]
        outputter = getattr(first_obj, keyword)
        string = outputter(header=True)
        if len(self.pyscal_list) > 1:
            for pyscal_obj in self.pyscal_list[1:]:
                outputter = getattr(pyscal_obj, keyword)
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

    def SGFN(self, write_to_filename=None):
        """Make SGFN string and optionally print to file"""
        return self.make_ecl_output("SGFN", write_to_filename)

    def SWFN(self, write_to_filename=None):
        """Make SWFN string and optionally print to file"""
        return self.make_ecl_output("SWFN", write_to_filename)

    def SOF3(self, write_to_filename=None):
        """Make SOF3 string and optionally print to file"""
        return self.make_ecl_output("SOF3", write_to_filename)

    def __len__(self):
        """Return the count of Pyscal objects in the list"""
        return len(self.pyscal_list)
