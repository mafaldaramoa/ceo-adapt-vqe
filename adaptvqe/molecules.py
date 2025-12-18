#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 09:23:22 2022

@author: mafal
"""
import numpy as np

from openfermion import MolecularData
from openfermionpyscf import run_pyscf


def create_h2(r):
    """
    Arguments:
        r (float): interatomic distance (angstrom)
    Returns:
        h2 (PyscfMolecularData): the linear H2 molecule at interatomic distance r, in the minimal STO-3G basis set
    """

    geometry = [['H', [0, 0, 0]], ['H', [0, 0, r]]]
    basis = 'sto-3g'
    multiplicity = 1
    charge = 0
    h2 = MolecularData(geometry, basis, multiplicity, charge, description='H2')
    h2 = run_pyscf(h2, run_fci=True, run_ccsd=True)

    return h2


def create_h3(r):
    """
    Arguments:
        r (float): interatomic distance (angstrom)
    Returns:
        h3 (PyscfMolecularData): the linear H3 molecule at interatomic distance r, in the minimal STO-3G basis set
    """

    geometry = [['H', [0, 0, 0]], ['H', [0, 0, r]], ['H', [0, 0, 2 * r]]]
    basis = 'sto-3g'
    multiplicity = 2  # odd number of electrons
    charge = 0
    h3 = MolecularData(geometry, basis, multiplicity, charge, description='H3')
    h3 = run_pyscf(h3, run_fci=True, run_ccsd=False)  # CCSD doesn't work here?

    return h3


def create_h4(r):
    """
    Arguments:
        r (float): interatomic distance (angstrom)
    Returns:
        h4 (PyscfMolecularData): the linear H4 molecule at interatomic distance r, in the minimal STO-3G basis set
    """

    geometry = [('H', (0, 0, 0)), ('H', (0, 0, r)), ('H', (0, 0, 2 * r)),
                ('H', (0, 0, 3 * r))]
    basis = 'sto-3g'
    multiplicity = 1
    charge = 0
    h4 = MolecularData(geometry, basis, multiplicity, charge, description='H4')
    h4 = run_pyscf(h4, run_fci=True, run_ccsd=True)

    return h4


def create_h5(r):
    """
    Arguments:
        r (float): interatomic distance (angstrom)
    Returns:
        h5 (PyscfMolecularData): the linear H5 molecule at interatomic distance r, in the minimal STO-3G basis set
    """

    geometry = [('H', (0, 0, 0)), ('H', (0, 0, r)), ('H', (0, 0, 2 * r)),
                ('H', (0, 0, 3 * r)), ('H', (0, 0, 4 * r))]
    basis = 'sto-3g'
    multiplicity = 2  # odd number of electrons
    charge = 0
    h5 = MolecularData(geometry, basis, multiplicity, charge, description='H5')
    h5 = run_pyscf(h5, run_fci=True, run_ccsd=False)  # CCSD doesn't work here?

    return h5


def create_h6(r):
    """
    Arguments:
        r (float): interatomic distance (angstrom)
    Returns:
        h6 (PyscfMolecularData): the linear H6 molecule at interatomic distance r, in the minimal STO-3G basis set
    """

    geometry = [('H', (0, 0, 0)), ('H', (0, 0, r)), ('H', (0, 0, 2 * r)),
                ('H', (0, 0, 3 * r)), ('H', (0, 0, 4 * r)), ('H', (0, 0, 5 * r))]
    basis = 'sto-3g'
    multiplicity = 1
    charge = 0
    h6 = MolecularData(geometry, basis, multiplicity, charge, description='H6')
    h6 = run_pyscf(h6, run_fci=True, run_ccsd=True)

    return h6


def create_h7(r):
    """
    Arguments:
        r (float): interatomic distance (angstrom)
    Returns:
        h7 (PyscfMolecularData): the linear H7 molecule at interatomic distance r, in the minimal STO-3G basis set
    """

    geometry = [('H', (0, 0, 0)), ('H', (0, 0, r)), ('H', (0, 0, 2 * r)),
                ('H', (0, 0, 3 * r)), ('H', (0, 0, 4 * r)), ('H', (0, 0, 5 * r)), ('H', (0, 0, 6 * r))]
    basis = 'sto-3g'
    multiplicity = 2  # odd number of electrons
    charge = 0
    h7 = MolecularData(geometry, basis, multiplicity, charge, description='H7')
    h7 = run_pyscf(h7, run_fci=True, run_ccsd=False)

    return h7


def create_h8(r):
    """
    Arguments:
        r (float): interatomic distance (angstrom)
    Returns:
        h8 (PyscfMolecularData): the linear H8 molecule at interatomic distance r, in the minimal STO-3G basis set
    """

    geometry = [('H', (0, 0, 0)), ('H', (0, 0, r)), ('H', (0, 0, 2 * r)), ('H', (0, 0, 3 * r)),
                ('H', (0, 0, 4 * r)), ('H', (0, 0, 5 * r)), ('H', (0, 0, 6 * r)),('H', (0, 0, 7*r))]
    basis = 'sto-3g'
    multiplicity = 1
    charge = 0
    h8 = MolecularData(geometry, basis, multiplicity, charge, description='H8')
    h8 = run_pyscf(h8, run_fci=True, run_ccsd=False)

    return h8


def create_triangular_h6(r):
    """
    Arguments:
        r (float): interatomic distance (angstrom)
    Returns:
        h6 (PyscfMolecularData): the triangular H6 molecule in the minimal
            STO-3G basis set. All atoms are separated by r. Within a equilateral
            triangle, we have 3 atoms in the vertices and 3 atoms in the center
            of each edge.
    """
    # The equilateral triangle can be divided into 4 smaller equilateral
    #triangles with side r. The height of these triangles is h and can be
    #calculating using Pythagoras theorem. This is also the y distance between
    #atoms with different y coordinates, assuming the basis of the triangular
    #is perpendicular to the y axis.
    h = r*np.sqrt(3)/2

    geometry = [('H', (0, 0, 0)),
                ('H', (-r/2, -h, 0)),
                ('H', (r/2, -h, 0)),
                ('H', (-r, -2*h, 0)),
                ('H', (0, -2*h, 0)),
                ('H', (r, -2*h, 0))]

    basis = 'sto-3g'
    multiplicity = 1
    charge = 0
    h6 = MolecularData(geometry, basis, multiplicity, charge, description='H6_trg')
    h6 = run_pyscf(h6, run_fci=True, run_ccsd=True)

    return h6

def create_lih(r):
    """
    Arguments:
        r (float): interatomic distance (angstrom)
    Returns:
        lih (PyscfMolecularData): the LiH molecule at interatomic distance r, in the minimal STO-3G basis set
    """

    geometry = [['Li', [0, 0, 0]], ['H', [0, 0, r]]]
    basis = 'sto-3g'
    multiplicity = 1
    charge = 0
    li_h = MolecularData(geometry, basis, multiplicity, charge, description='LiH')
    li_h = run_pyscf(li_h, run_fci=True, run_ccsd=True)

    return li_h


def create_beh2(r):
    """
    Arguments:
        r (float): interatomic distance (angstrom)
    Returns:
        beh2 (PyscfMolecularData): the BeH2 molecule at interatomic distance r, in the minimal STO-3G basis set
    """

    geometry = [['H', [0, 0, 0]], ['Be', [0, 0, r]], ['H', [0, 0, 2 * r]]]
    basis = 'sto-3g'
    multiplicity = 1
    charge = 0
    be_h2 = MolecularData(geometry, basis, multiplicity, charge, description='BeH2')
    be_h2 = run_pyscf(be_h2, run_fci=True, run_ccsd=True)

    return be_h2
