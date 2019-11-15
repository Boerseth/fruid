# Simulating 2D fluids numerically

I just googled "Navier-Stokes", and have tried to come up with a scheme to numerically integrate the PDE that I found. Will eventually upload a LaTeX document that explains the mathematics at least.

The 2D system consists of three functions in x,y, all of which have time derivatives that are nonlinearly dependent on the other two. Coming up with a nice way to represent this programatically has been the first step, and I have written classes that handle the different time derivatives, even though the explicit mathematical details are not yet implemented.

I have also made plans on how to implement obstacles for the fluid flow, i.e. disallowed regions within the 2D landscape, by masking off the corresponding rows and columns in vectors and matrices, and applying appropriate boundary conditions. The second part of that, with respect to all the different spatial derivatives that are needed in the formulae, is definitely what will be the trickiest in all this to get right.
