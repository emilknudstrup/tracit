.. tracit documentation master file, created by
   sphinx-quickstart on Fri Feb  4 08:08:28 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

:strike:`tracit`
=================================

**tracing the paths of transiting exoplanets**

:strike:`tracit`'s aim is to shed light on key orbital parameters in exoplanet systems.

This is done through modelling of light curves and radial velocity curves, but :strike:`tracit` is specifically designed with the Rossiter-McLaughlin effect in mind.


References and acknowledgements
-------------------------------

The planetary shadow is modelled following the approach in:
   * `Albrecht et al. (2007) <https://ui.adsabs.harvard.edu/abs/2007A%26A...474..565A/abstract>`_
   
The RM effect as seen in the RVs is modelled using the software presented in:
   * `Hirano et al. (2011) <https://ui.adsabs.harvard.edu/abs/2011ApJ...742...69H/abstract>`_

:strike:`tracit` has been used in:
   * `Knudstrup & Albrecht (2021) <https://ui.adsabs.harvard.edu/abs/2021arXiv211114968K/abstract>`_

.. toctree::
   :maxdepth: 1
   :caption: Usage

   usage

.. toctree::
   :maxdepth: 2
   :caption: API
   
   API/expose
   API/shady
   API/business
   API/dynamics



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
