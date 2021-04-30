Cherab MAST-U demo
==================

Minimal working demo for using the Cherab spectroscopy package to produce synthetic images of MAST-U, based on the
output of a given SOLPS simulation. A lot of this code will be ropey.

Uses a toy pinhole camera model that is specified by its pupil position and forward vector in Cartesian space (units m, 
origin at the centre of the machine.)

I have tested this with the versions of Raysect (0.6.0) and Cherab (1.2.0) I was using last summer and it works. 
TODO: test with current versions.