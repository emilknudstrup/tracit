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
   * :cite:t:`Albrecht2007`
   * :cite:t:`Albrecht2013`

The subplanetary velocities are modelled following the approach in:
   * :cite:t:`Albrecht2011`

The RM effect as seen in the RVs is modelled using the software presented in:
   * :cite:t:`Hirano2011`

:strike:`tracit` has been used in:
   * :cite:t:`Hjorth2021`
   * :cite:t:`Knudstrup2021`

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
   :maxdepth: 1
   :caption: API
   
   API/expose
   API/shady
   API/business
   API/structure
   API/support
   API/dynamics
   API/priors

.. toctree::
   :maxdepth: 2
   :caption: Data preparation

   data
   CCF/shazam

.. toctree::
   :maxdepth: 1
   :caption: References

   references

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
