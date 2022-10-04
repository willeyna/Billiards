# Billiard Bounces within 2D Smooth Manifold Boundaries 

This project began Jan 2022 as a simple curiosity-founded personal project to analyze bounded 2D collisions and the patterns or chaotic motion that may be produced.
Prior to starting work on this I had not read any literature on the topic and have approached this as a 'discorvery' project,  building algorithms used at every step. As such, methods used may be clearly sub-optimal. Instead of optimization, this project is focused on tinkering and discovery of numerical methods.

In its current iteration velocity is updated each iteration using central differences to approximate the normal on the boundary and positions are updated by performing 1D minimization of distance to boundary along velocity line using a bracketing method. 

Goals of the project: 
  * Create script that can model a sequence of N bounces given any differentiable boundary f(x,y) with derivative dy/dx = f'(x)
  * Plot and qualitatively analyze the motion of billiards in basic circular and elliptical boundaries 
  * Understand and analyze poincare maps of the motion to understand chaotic or non-chaotic motion
  * Analyze all such parts for more complex boundaries like cardiods 
  

