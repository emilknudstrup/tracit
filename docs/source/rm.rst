.. _The RM effect:

The Rossiter-McLaughlin effect 
===============================

The Rossiter-McLaughlin (RM) effect is a distortion of the stellar line shapes caused by a transiting body blocking part of the rotating stellar disk. This effect allows us to measure :math:`\lambda`, the sky projection of the stellar obliquity, :math:`\psi`.

.. figure:: figures/rm_geo.pdf
	:name: rm_geo

	The angles between the stellar spin axis, the orbital axis of a planet, and the observer. 

The geometry of the problem is shown in :numref:`rm_geo`, where :math:`\hat{n}_{\rm obs}`, :math:`\hat{n}_{\rm o}`, and :math:`\hat{n}_\star` are  the unit vectors for the observer, the orbital angular momentum, and the stellar angular momentum.


Summarized below are the three different approaches :strike:`tracit` can be used to model the RM effect. 

Planetary shadow
---------------------------

To model the planet shadow we first construct a limb-darkened stellar grid. In our models we assume a quadratic limb-darkening law of the form

.. math::
	I = 1 - u_1(1 - \mu) - u_2 (1 - \mu)^2 \, .

Here :math:`\mu=\cos \theta` with :math:`\theta` being the angle between the local normal and a line parallel to the line of sight, and :math:`I` is the local normalized by the intensity at the center of the disc, i.e., :math:`\mu=1`. :math:`u_1` and :math:`u_2` are the linear and quadratic limb-darkening coefficients, respectively. 

Assuming solid body rotation, the radial velocity of the stellar surface is a function of the distance from the stellar spin axis only. If the :math:`x`-axis is along the stellar equator and the :math:`y`-axis parallel to the projected stellar spin axis, then the projected stellar rotation speed at :math:`x` is simply

.. math::
	v_\mathrm{p}=\frac{x}{R} v \sin i \, ,

where :math:`R` is the stellar radius, and :math:`v \sin i` the projected stellar rotation speed. The Doppler velocity of the stellar surface below a planet at :math:`x` is thus :math:`v_\mathrm{p}`.

In :strike:`tracit` we then calculate for each time stamp the position of the planet, and if the planet is inside the stellar disk we set the intensity of the pixels blocked by the planet to zero. In each of the pixels in the limb-darkened, partially obscured stellar grid, the effects of macro-, :math:`\zeta`, and microturbulence, :math:`\xi`, are then accounted for following the approach in :cite:t:`Gray2005`. 

Subplanetary velocities
---------------------------

As argued above :math:`v_\mathrm{p}` only depends on the :math:`x` coordinate and should therefore progress linearly with time. We can therefore calculate :math:`v_\mathrm{p}` with a first order polynoimial with extremes occuring at :math:`v_\mathrm{ingress}` and :math:`v_\mathrm{egress}`. The offset and slope of the line are given by

.. math::
	v_\mathrm{egress} - v_\mathrm{ingress} = 2 \times (v \sin i) \sin \lambda \times b \, ,
.. math::
	v_\mathrm{egress} + v_\mathrm{ingress} = 2 \times (v \sin i) \sin \lambda \times \sqrt{1 - b^2} \, .


Radial velocities
---------------------------

The distortion of the stellar lines lead to anomalous radial velocities (RVs) observed during transit. A first-order estimate of the anomalous stellar RVs can be obtained from

.. math::
	\mathrm{RV_{RM}} (t) \approx - \frac{r}{R} v_\mathrm{p}(t) \, .

The :math:`\mathrm{RV_{RM}}` measurements relate to :math:`v_\mathrm{p}` and the radius ratio of the transiting to the occulted object, :math:`r/R`. The sign change occurs as the subplanetary light is blocked from view. Any particular :math:`\mathrm{RV_{RM}}` is further modified by the stellar limb dark.