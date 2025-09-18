#!/usr/bin/env python
# coding: utf-8

# In[1]:


from   finder import mag_detection_curve, surface_detection_curve
import matplotlib.pyplot as plt
import astropy.cosmology as cosmology
import numpy as np

cosmo = cosmology.FlatLambdaCDM(70, 0.3)
z     = np.linspace(1, 4, 100)

scrit = surface_detection_curve(z, cosmo)
mcrit = mag_detection_curve(z, cosmo)

f  = plt.figure(figsize=(6, 6))
ax = f.add_subplot(111)
ax.tick_params(direction='in', top=True)

plt.plot(z, mcrit, color='k')
plt.ylabel('Flux detection curve [AB mag]')

# Showing surface criterion in 1e-3 arcsec^2
ax2 = ax.twinx()
ax2.tick_params(direction='in', right=True, left=False, color='firebrick', labelcolor='firebrick')
ax2.spines['right'].set_color('firebrick')

plt.plot(z, scrit * 1e3, color='firebrick', ls='--')
plt.ylabel('Surface detection curve [$10^{-3}$ arcsec$^2$]', color='firebrick')
plt.xlabel('Redshift')

plt.xlim(1, 4)
plt.show()

