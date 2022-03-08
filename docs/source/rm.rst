.. _The RM effect:

The Rossiter-McLaughlin effect 
===============================

The Rossiter-McLaughlin (RM) effect is a distortion of the stellar line shapes caused by a transiting body blocking part of the rotating stellar disk. This effect allows us to measure :math:`\lambda`, the sky projection of the stellar obliquity, :math:`\psi`.

Summarized below are the three different approaches :strike:`tracit` can be used to model the RM effect. 

Planetary shadow
---------------------------

To model the planet shadow we first construct a limb-darkened stellar grid. In our models we assume a quadratic limb-darkening law of the form


.. math::
	I = 1 - u_1(1 - \mu) - u_2 (1 - \mu)^2 \, .

Here :math:`\mu=\cos \theta` with :math:`\theta` being the angle between the local normal and a line parallel to the line of sight, and :math:`I` is the local normalized by the intensity at the center of the disc, i.e., :math:`\mu=1`.

Subplanetary velocities
---------------------------

.. math::
	V_\mathrm{egress} - V_\mathrm{ingress} = 2 \times (v \sin i) \sin \lambda \times b \, ,
.. math::
	V_\mathrm{egress} + V_\mathrm{ingress} = 2 \times (v \sin i) \sin \lambda \times \sqrt{1 - b^2} \, .


Radial velocities
---------------------------

The distortion of the stellar lines lead to anomalous radial velocities (RVs) observed during transit. A first-order estimate of the anomalous stellar RVs can be obtained from

.. math::
	\mathrm{RV_{RM}} (t) \approx - \frac{r}{R} v_\mathrm{p}(t) \, .

The :math:`\mathrm{RV_{RM}}` measurements relate to :math:`v_\mathrm{p}` and the radius ratio of the transiting to the occulted object, :math:`r/R`. The sign change occurs as the subplanetary light is blocked from view. Any particular :math:`\mathrm{RV_{RM}}` is further modified by the stellar limb dark.