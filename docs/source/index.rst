.. tracit documentation master file, created by
   sphinx-quickstart on Fri Feb  4 08:08:28 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

:strike:`tracit`
=================================

**tracing the paths of transiting exoplanets**

:strike:`tracit`'s aim is to shed light on key orbital parameters in exoplanet systems.

This is done through modelling of light curves and radial velocity curves. However, :strike:`tracit` is specifically designed with the Rossiter-McLaughlin effect in mind, be it through the planetary shadow, subplanetary velocities, or radial velocities.


References and acknowledgements
-------------------------------

The planetary shadow is modelled following the approaches in:
   * `Albrecht et al. (2007) <https://ui.adsabs.harvard.edu/abs/2007A%26A...474..565A/abstract>`_
   * `Albrecht et al. (2013) <https://ui.adsabs.harvard.edu/abs/2013ApJ...771...11A/abstract>`_

The subplanetary velocities are modelled following the approach in:
   * `Albrecht et al. (2011) <https://ui.adsabs.harvard.edu/abs/2011ApJ...738...50A/abstract>`_

The RM effect as seen in the RVs is modelled using the software presented in:
   * `Hirano et al. (2011) <https://ui.adsabs.harvard.edu/abs/2011ApJ...742...69H/abstract>`_

:strike:`tracit` has been used in:
   * `Hjorth et al. (2021) <https://ui.adsabs.harvard.edu/abs/2021PNAS..11820174H/abstract>`_
   * `Knudstrup & Albrecht (2021) <https://ui.adsabs.harvard.edu/abs/2021arXiv211114968K/abstract>`_

.. toctree::
   :maxdepth: 1
   :caption: Usage

   usage
   examples/hd332231/fit.ipynb
   examples/shadow_hd332231/fit.ipynb

.. toctree::
   :maxdepth: 1
   :caption: The RM effect

   rm

.. toctree::
   :maxdepth: 2
   :caption: API
   
   API/expose
   API/shady
   API/business
   API/structure
   API/dynamics

.. toctree::
   :maxdepth: 2
   :caption: CCF

   CCF/shazam

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
