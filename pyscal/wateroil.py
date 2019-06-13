# -*- coding: utf-8 -*-

import math
import copy
import numpy as np
import pandas as pd

from pyscal.constants import EPSILON as epsilon
from pyscal.constants import SWINTEGERS

class WaterOil(object):
    """A representation of two-phase properties for oil-water.

    Can hold relative permeability data, and capillary pressure.
   
    Parametrizations for relative permeability:
     * Corey
     * LET

    For capillary pressure:
     * Simplified J-function

    For object initialization, only saturation endpoints must be inputted,
    and saturation resolution. An optional string can be added as a 'tag'
    that can be used when outputting.
    
    Relative permeability and/or capillary pressure can be added through
    parametrizations, or from a dataframe (will incur interpolation).

    Can be dumped as include files for Eclipse/OPM and Nexus simulators.
    """

    def __init__(self, swirr=0.0, swl=0.0, swcr=0.0, sorw=0.0, h=0.01, tag=""):
        """Sets up the saturation range. Swirr is only relevant
        for the capillary pressure, not for relperm data."""

        assert swirr < 1.0 + epsilon
        assert swirr > - epsilon
        assert swl < 1.0 + epsilon
        assert swl > - epsilon
        assert swcr > - epsilon
        assert swcr < 1.0 + epsilon
        assert sorw > - epsilon
        assert sorw < 1.0 + epsilon
        assert h > epsilon
        assert h < 1
        assert swl < 1 - sorw
        assert swcr < 1 - sorw
        assert swirr < 1 - sorw

        self.swirr = swirr
        self.swl = max(swl, swirr)  # Cannot allow swl < swirr. Warn?
        self.swcr = max(self.swl, swcr)  # Cannot allow swcr < swl. Warn?
        self.sorw = sorw
        self.h = h
        self.tag = tag
        sw = list(np.arange(self.swl, 1-sorw, h)) + \
             [self.swcr] + [1 - sorw] + [1]
        self.table = pd.DataFrame(sw, columns=['sw'])
        self.table['swint'] = list(map(int, list(map(round,
                                           self.table['sw'] * SWINTEGERS))))
        self.table.drop_duplicates('swint', inplace=True)
        self.table.sort_values(by='sw', inplace=True)
        self.table.reset_index(inplace=True)
        self.table = self.table[['sw']]  # Drop the swint column
        self.table['swn'] = (self.table.sw - self.swcr) \
                            / (1 - self.swcr - sorw)  # Normalize
        self.table['son'] = (1 - self.table.sw - sorw) \
                            / (1 - swl - sorw)  # Normalized

        # Different normalization for Sw used for capillary pressure
        self.table['swnpc'] = (self.table.sw - swirr) / (1 - swirr)

        self.swcomment = "-- swirr=%g swl=%g swcr=%g sorw=%g\n" \
                         % (self.swirr, self.swl, self.swcr, self.sorw)
        self.krwcomment = ""
        self.krowcomment = ""
        self.pccomment = ""

    def add_oilwater_fromtable(self, df, swcolname='Sw',
                               krwcolname='krw', krowcolname='krow',
                               pccolname='pcow', krwcomment="",
                               krowcomment="", pccomment=""):
        """Interpolate relpermdata from a dataframe.

        The saturation range with endpoints must be set up beforehand,
        and must be compatible with the tabular input. The tabular
        input will be interpolated to the initialized Sw-table

        If you have krw and krow in different dataframes, call this
        function twice

        Calling function is responsible for checking if any data was
        actually added to the table.

        The python package ecl2df has a tool for converting Eclipse input
        files to dataframes. 
        
        Args:
            df: Pandas dataframe containing data
            swcolname: string, column name with the saturation data in the dataframe.
            krwcolname: string, name of the column with krw
            krowcolname: string
            pccolname: string
            krwcomment: string
            krowcomment: string
            pccomment: string
        """
        from scipy.interpolate import PchipInterpolator
        if swcolname not in df:
            raise Exception(swcolname +
                            " not found in dataframe, can't read table data")
        if krwcolname in df:
            pchip = PchipInterpolator(df[swcolname].astype(float),
                                      df[krwcolname].astype(float))
            self.table['krw'] = pchip(self.table.sw)
            self.krwcomment = "-- krw from tabular input" + krwcomment + "\n"
        if krowcolname in df:
            pchip = PchipInterpolator(df[swcolname].astype(float),
                                      df[krowcolname].astype(float))
            self.table['krow'] = pchip(self.table.sw)
            self.krowcomment = "-- krow from tabular input" + krowcomment + "\n"
        if pccolname in df:
            pchip = PchipInterpolator(df[swcolname].astype(float),
                                      df[pccolname].astype(float))
            self.table['pc'] = pchip(self.table.sw)
            self.pccomment = "-- pc from tabular input" + pccomment + "\n"

    def add_corey_water(self, nw=2, krwend=1, krwmax=1):
        """ Add krw data through the Corey parametrization

        A column named 'krw' will be added. If it exists, it will
        be replaced.
           
        It is assumed that there are no sw points between
        sw=1-sorw and sw=1, which should give linear 
        interpolations in simulators. The corey parameter
        applies up to 1-sorw.

        Args:
            nw: float, Corey parameter for water.
            krwend: float, value of krw at 1 - sorw.
            krwmax: float, maximal value at Sw=1

        """
        assert nw > 0
        assert krwend < 2
        assert krwend > 0
        assert krwmax < 2
        assert krwmax > 0
        self.table['krw'] = krwend * self.table.swn ** nw
        self.table.loc[self.table.sw > (1 - self.sorw + epsilon), 'krw'] \
            = max(krwmax, krwend)
        self.table.loc[self.table.sw < self.swcr, 'krw'] = 0
        self.krwcomment = "-- Corey krw, nw=%g, krwend=%g, krwmax=%g\n" \
                          % (nw, krwend, krwmax)

    def add_LET_water(self, l=2, e=2, t=2, krwend=1, krwmax=1):
        """Add krw data through LET parametrization

        It is assumed that there are no sw points between
        sw=1-sorw and sw=1, which should give linear 
        interpolations in simulators. The LET parameters
        apply up to 1-sorw.

        Args:
            l: float
            e: float
            t: float
            krwend: float
            krwmax: float
        """
        self.table['krw'] = krwend * self.table.swn ** l \
                         / ((self.table.swn ** l) \
                            + e * (1 - self.table.swn) ** t)
        # Be careful here, because we want only Sw points at 1-sorw and 1
        # between those values there should be no Sw, because we want
        # linear interpolation in that range.
        self.table.loc[self.table.sw > (1 - self.sorw - epsilon), 'krw'] \
            = max(krwmax, krwend)
        self.table.loc[np.isclose(self.table.sw, 1 - self.sorw), 'krw'] \
            = krwend
        self.table.loc[self.table.sw < self.swcr, 'krw'] = 0
        self.krwcomment \
            = "-- LET krw, l=%g, e=%g, t=%g, krwend=%g, krwmax=%g\n" \
            % (l, e, t, krwend, krwmax)

    def add_LET_oil(self, l=2, e=2, t=2, kroend=1, kromax=1):
        """
        Add kro data through LET parametrization

        Args:
            l: float
            e: float
            t: float
            kroend: float
            kromax: float
        """
        self.table['krow'] = kroend * self.table.son ** l \
                          / ((self.table.son ** l) \
                             + e * (1 - self.table.son) ** t)
        self.table.loc[self.table.sw >= (1 - self.sorw), 'krow'] = 0
        self.table.loc[self.table.sw < self.swl, 'krow'] = kromax
        self.krowcomment \
            = "-- LET krow, l=%g, e=%g, t=%g, kroend=%g, kromax=%g\n" \
                           % (l, e, t, kroend, kromax)

    def add_corey_oil(self, now=2, kroend=1, kromax=1):
        """Add kro data through the Corey parametrization,
        paying attention to saturations above sorw and below swl"""
        assert now > 0
        assert kroend < 2
        assert kromax < 2
        self.table['krow'] = kroend * self.table.son ** now
        self.table.loc[self.table.sw >= (1 - self.sorw), 'krow'] = 0
        self.table.loc[self.table.sw < self.swl, 'krow'] = kromax
        self.krowcomment = "-- Corey krow, now=%g, kroend=%g, kromax=%g\n" \
                           % (now, kroend, kromax)

    def add_simple_J(self, a=5, b=-1.5,
                     poro_ref=0.25, perm_ref=100, drho=300, g=9.81):
        """Add capillary pressure function from a simplified J-function

        This is the 'inverse' or 'RMS' version of the a and b, the formula
        is

            J = a S_w^b

        J is not dimensionless.
        Doc: https://wiki.equinor.com/wiki/index.php/Res:Water_saturation_from_Leverett_J-function

        poro_ref is a fraction, between 0 and 1
        perm_ref is in milliDarcy
        drho has SI units kg/m³. Default value is 300
        g has SI units m/s², default value is 9.81
        """
        # drho = rwo_w - rho_o, in units g/cc

        # swnpc is a normalized saturation, but normalized with
        # respect to swirr, not to swl (the swirr here is sometimes
        # called 'swirra' - asymptotic swirr)
        self.table['J'] = a * self.table.swnpc ** b
        self.table['H'] = self.table.J * math.sqrt(poro_ref / perm_ref)
        # Scale drho and g from SI units to g/cc and m/s²100
        self.table['pc'] = self.table.H * drho/1000 * g/100.0
        self.pccomment = "-- Simplified J function for Pc; \n--   " \
            + "a=%g, b=%g, poro_ref=%g, perm_ref=%g mD, drho=%g kg/m³, g=%g m/s²\n" \
            % (a, b, poro_ref, perm_ref, drho, g)

    def add_skjaeveland_pc(self, cw, co, aw, ao, swr=None, sor=None):
        """Add capillary pressure from the Skjæveland correlation,

        Doc: https://wiki.equinor.com/wiki/index.php/Res:The_Skjaeveland_correlation_for_capillary_pressure

        The implementation is unit independent, units are contained in the
        input constants.

        If swr and sor are not provided, it will be taken from the
        swirr and sorw. Only use different values here if you know
        what you are doing.

        Modifies or adds self.table.pc if succesful.
        Returns false if error occured.

        """
        inputerror = False
        if cw < 0:
            print("cw must be larger or equal to zero")
            inputerror = True
        if co > 0:
            print("co must be less than zero")
            inputerror = True
        if aw <= 0:
            print("aw must be larger than zero")
            inputerror = True
        if ao <= 0:
            print("ao must be larger than zero")
            inputerror = True

        if swr == None:
            swr = self.swirr
        if sor == None:
            sor = self.sorw

        if swr >= 1 - sor:
            print("swr (swirr) must be less than 1 - sor")
            inputerror = True
        if swr < 0 or swr > 1:
            print("swr must be contained in [0,1]")
            inputerror = True
        if sor < 0 or sor > 1:
            print("sor must be contained in [0,1]")
            inputerror = True
        if inputerror:
            return False
        self.pccomment = "-- Skjæveland correlation for Pc;\n"\
            + "-- cw=%g, co=%g, aw=%g, ao=%g, swr=%g, sor=%g\n" \
            % (cw, co, aw, ao, swr, sor)

        # swnpc is a normalized saturation, but normalized with
        # respect to swirr, not to swl (the swirr here is sometimes
        # called 'swirra' - asymptotic swirr)

        # swnpc is generated upon object initialization, but overwritten
        # here to most likely the same values.
        self.table['swnpc'] = (self.table.sw - swr) / (1 - swr)

        # sonpc is almost like 'son', but swl is not used here:
        self.table['sonpc'] = (1 - self.table.sw - sor)/(1 - sor)

        # The Skjæveland correlation
        self.table.loc[self.table.sw < 1 - sor, 'pc'] \
             = cw / (self.table.swnpc ** aw) + \
             co / (self.table.sonpc ** ao)

        # From 1-sor, the pc is not defined. We want to extrapolate constantly,
        # but with a twist as Eclipse does not non-monotone capillary pressure:
        self.table['pc'].fillna(value=self.table.pc.min(), inplace=True)
        nanrows = self.table.sw > 1 - sor - epsilon
        self.table.loc[nanrows, 'pc'] = self.table.loc[nanrows, 'pc'] - \
            self.table.loc[nanrows, 'sw'] # Just deduct sw to make it monotone..


    def add_LET_pc_pd(self, Lp, Ep, Tp, Lt, Et, Tt, Pcmax, Pct):
        """Add a primary drainage LET capillary pressure curve.

        Docs: https://wiki.equinor.com/wiki/index.php/Res:The_LET_correlation_for_capillary_pressure

        Note that Pc where Sw > 1 - sorw will appear linear because
        there are no saturation points in that interval.
        """

        # The "forced part"
        self.table['Ffpcow'] = (1 - self.table.swnpc) ** Lp / \
            ((1 - self.table.swnpc) ** Lp + Ep * self.table.swnpc ** Tp)

        # The gradual rise part:
        self.table['Ftpcow'] = self.table.swnpc ** Lt / \
            (self.table.swnpc ** Lt + Et * (1 - self.table.swnpc) ** Tt)

        # Putting it together:
        self.table['pc'] = (Pcmax - Pct) * self.table.Ffpcow \
                           - Pct * self.table.Ftpcow \
                           + Pct

        # Special handling of the interval [0,swirr]
        self.table.loc[self.table.swn < epsilon, 'pc'] = Pcmax
        self.pccomment = "-- LET correlation for primary drainage Pc;\n"\
            + "-- Lp=%g, Ep=%g, Tp=%g, Lt=%g, Et=%g, Tt=%g, " +\
            "Pcmax=%g, Pct=%g\n" \
            % (Lp, Ep, Tp, Lt, Et, Tt, Pcmax, Pct)

    def add_LET_pc_imb(self, Ls, Es, Ts, Lf, Ef, Tf, Pcmax, Pcmin, Pct):
        """Add an imbition LET capillary pressure curve.

        Docs: https://wiki.equinor.com/wiki/index.php/Res:The_LET_correlation_for_capillary_pressure
        """

        # Normalized water saturation including sorw
        self.table['swnpco'] = (self.table.sw - self.swirr) \
                               / (1 - self.sorw - self.swirr)

        # The "forced part"
        self.table['Fficow'] = (1 - self.table.swnpco) ** Ls / \
            ((1 - self.table.swnpco) ** Ls + Es * self.table.swnpco ** Ts)

        # The gradual rise part:
        self.table['Fticow'] = self.table.swnpco ** Lf / \
            (self.table.swnpco ** Lf + Ef * (1 - self.table.swnpco) ** Tf)

        # Putting it together:
        self.table['pc'] = (Pcmax + Pct) * self.table.Fficow \
                           - (Pcmin - Pct) * self.table.Fticow - Pct

        # Special handling of the interval [0,swirr]
        self.table.loc[self.table.swnpco < epsilon, 'pc'] = Pcmax
        # and [1-sorw,1]
        self.table.loc[self.table.swnpco > 1 - epsilon, 'pc'] \
            = self.table.pc.min()
        self.pccomment = "-- LET correlation for imbibition Pc;\n"\
            + "-- Ls=%g, Es=%g, Ts=%g, Lf=%g, Ef=%g, Tf=%g, " +\
            "Pcmax=%g, Pcmin=%g, Pct=%g\n" \
            % (Ls, Es, Ts, Lf, Ef, Tf, Pcmax, Pcmin, Pct)

    def selfcheck(self):
        """Check validities of the data in the table.

        If you call SWOF, this function must not return False
        """
        error = False
        if not (self.table.sw.diff().dropna().round(10) > - epsilon).all():
            print("Error: sw data not strictly increasing")
            error = True
        if not (self.table.krw.diff().dropna().round(10) >= - epsilon).all():
            print("Error: krw data not monotonely increasing")
            error = True
        if 'krow' in self.table.columns and \
                not (self.table.krow.diff().dropna().round(10)
                     <= epsilon).all():
            print("Error: krow data not monotonely decreasing")
            error = True
        if 'pc' in self.table.columns and self.table.pc[0] >  - epsilon:
            if not (self.table.pc.diff().dropna().round(10) < epsilon).all():
                print("Error: pc data not strictly decreasing")
                error = True
        if 'pc' in self.table.columns and np.isinf(self.table.pc.max()):
            print("Error: pc goes to infinity. Maybe swirr=swl?")
            error = True
        for col in list(set(['sw', 'krw', 'krow']) & set(self.table.columns)):
            if not ((round(min(self.table[col]), 10) >= - epsilon) and
                    (round(max(self.table[col]), 10) <= 1 + epsilon)):
                print("Error: %s data should be contained in [0,1]" %col)
                error = True
        if error:
            return False
        else:
            return True


    def SWOF(self, header=True, dataincommentrow=True):
        if not self.selfcheck():
            return
        string = ""
        if 'pc' not in self.table.columns:
            self.table['pc'] = 0
            self.pccomment = "-- Zero capillary pressure\n"
        if header:
            string += "SWOF\n"
        string += "-- " + self.tag + "\n"
        string += "-- Sw Krw Krow Pc\n"
        if dataincommentrow:
            string += self.swcomment
            string += self.krwcomment
            string += self.krowcomment
            string += "-- krw = krow @ sw=%1.5f\n" % self.crosspoint()
            string += self.pccomment
        string += self.table[['sw', 'krw', 'krow', 'pc']].\
                  to_csv(sep=' ', float_format='%1.7f',
                         header=None, index=False)
        string += "/\n" # Empty line at the end
        return string

    def SWFN(self, header=True, dataincommentrow=True):
        if not self.selfcheck():
            return
        string = ""
        if 'pc' not in self.table.columns:
            self.table['pc'] = 0
            self.pccomment = "-- Zero capillary pressure\n"
        if header:
            string += "SWFN\n"
        string += "-- " + self.tag + "\n"
        string += "-- Sw Krw Pc\n"
        if dataincommentrow:
            string += self.swcomment
            string += self.krwcomment
            if 'krow' in self.table.columns:
                string += "-- krw = krow @ sw=%1.5f\n" % self.crosspoint()
            string += self.pccomment
        string += self.table[['sw', 'krw', 'pc']].\
                  to_csv(sep=' ', float_format='%1.7f',
                         header=None, index=False)
        string += "/\n"  # Empty line at the end
        return string

    def WOTABLE(self, header=True, dataincommentrow=True):
        """Return a string for a Nexus WOTABLE"""
        string = ""
        if 'pc' not in self.table.columns:
            self.table['pc'] = 0
            self.pccomment = "-- Zero capillary pressure\n"
        if header:
            string += "WOTABLE\n"
            string += "SW KRW KROW PC\n"
        if dataincommentrow:
            string += self.swcomment.replace('--', '!')
            string += self.krwcomment.replace('--', '!')
            string += self.krowcomment.replace('--', '!')
            string += "! krw = krow @ sw=%1.5f\n" % self.crosspoint()
            string += self.pccomment.replace('--', '!')
        string += self.table[['sw', 'krw', 'krow', 'pc']]\
                      .to_csv(sep=' ', float_format='%1.7f',
                              header=None, index=False)
        return string

    def plotpc(self, ax=None, color='blue', alpha=1, label=None,
               linewidth=1, linestyle='-', logscale=False):
        """Plot capillary pressure (pc) a supplied matplotlib axis"""
        import matplotlib.pyplot as plt
        import matplotlib
        if not ax:
            matplotlib.style.use('ggplot')
            fig, useax = plt.subplots()
        else:
            useax = ax
        self.table.plot(ax=useax, x='sw', y='pc', c=color, alpha=alpha,
                        legend=None, linewidth=linewidth, linestyle=linestyle)
        if logscale:
            useax.set_yscale('log')
        if not ax:
            plt.show()

    def plotkrwkrow(self, ax=None, color='blue', alpha=1, label=None,
                    linewidth=1, linestyle='-'):
        """Plot krw and krow on a supplied matplotlib axis"""
        import matplotlib.pyplot as plt
        import matplotlib
        if not ax:
            matplotlib.style.use('ggplot')
            fig, useax = plt.subplots()
        else:
            useax = ax
        self.table.plot(ax=useax, x='sw', y='krw', c=color, alpha=alpha,
                        legend=None, linewidth=linewidth, linestyle=linestyle)
        self.table.plot(ax=useax, x='sw', y='krow', c=color, alpha=alpha,
                        legend=None, linewidth=linewidth, linestyle=linestyle)
        if not ax:
            plt.show()

    def crosspoint(self):
        """Locate and return the saturation point where krw = krow

        Accuracy of this crosspoint depends on the resolution chosen
        when initializing the saturation range
       """

        # Make a copy for calculations
        tmp = pd.DataFrame(self.table[['sw', 'krw', 'krow']])
        tmp.loc[:, 'krwminuskrow'] = tmp['krw'] - tmp['krow']

        # Add a zero value for the difference column, and interpolate
        # the sg column to the zero value
        zerodf = pd.DataFrame(index=[len(tmp)], data={'krwminuskrow' : 0.0})
        tmp = pd.concat([tmp, zerodf], sort=True)
        # When Pandas is upgraded for all users:
        #tmp = pd.concat([tmp, zerodf], sort=True)
        tmp.set_index('krwminuskrow', inplace=True)
        tmp.interpolate(method='slinear', inplace=True)

        return tmp[np.isclose(tmp.index, 0.0)].sw.values[0]
