# Multivariable Calculus

_Foundation Track A - Course 2_

## Course Overview

**Duration:** 3-4 months  
**Prerequisites:** Single Variable Calculus  
**Programming:** Python with NumPy  
**Time Commitment:** 12-15 hours/week

## Learning Path Structure

🔵 Core Content (Essential - Master These)  
🟡 Recommended (Deepen Understanding)  
🟢 Advanced (Extra Challenge)  
⭐ Applications (Real-world Usage)

## Unit 1: Vectors and Vector-Valued Functions (3 weeks)

### Learning Objectives

- Master vector operations in R² and R³
- Understand vector-valued functions
- Work with parametric curves and surfaces
- Implement vector computations
- Visualize vector fields and curves

### Topics

🔵 **Foundations**

1. Vectors in Space

   - Vector algebra
   - Dot product
   - Cross product
   - Lines and planes
   - Distance problems
   - Vector projections

2. Vector-Valued Functions
   - Parametric curves
   - Limits and continuity
   - Derivatives
   - Integration
   - Arc length
   - Curvature
3. Motion in Space
   - Position vectors
   - Velocity
   - Acceleration
   - Tangential/normal components
   - Angular velocity
   - Planetary motion

🟡 **Deeper Understanding**

1. Advanced Vector Operations

   - Triple products
   - Change of basis
   - Linear independence
   - Vector spaces preview
   - Quaternions introduction

2. Differential Geometry Preview
   - Frenet frame
   - Torsion
   - Osculating plane
   - Natural equations

🟢 **Theoretical Exploration**

1. Abstract Concepts
   - Vector spaces
   - Linear transformations
   - Differential forms preview
   - Tensors introduction

### Programming Implementations

```python
from typing import Callable, Union, Tuple
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class VectorAnalysis:
    """Tools for vector analysis and visualization."""

    def __init__(self):
        self.fig = plt.figure(figsize=(10, 10))
        self.ax = self.fig.add_subplot(111, projection='3d')

    def plot_vector_field(self, F: Callable[[np.ndarray], np.ndarray],
                         xlim: Tuple[float, float] = (-5, 5),
                         ylim: Tuple[float, float] = (-5, 5),
                         zlim: Tuple[float, float] = (-5, 5),
                         density: int = 10):
        """Plot 3D vector field."""
        x, y, z = np.meshgrid(np.linspace(*xlim, density),
                             np.linspace(*ylim, density),
                             np.linspace(*zlim, density))

        # Evaluate vector field
        points = np.stack([x, y, z])
        U, V, W = F(points)

        # Plot
        self.ax.quiver(x, y, z, U, V, W)
        plt.show()

    def plot_parametric_curve(self, r: Callable[[float], np.ndarray],
                            t_range: Tuple[float, float],
                            num_points: int = 1000):
        """Plot parametric curve in 3D."""
        t = np.linspace(*t_range, num_points)
        points = np.array([r(ti) for ti in t])

        self.ax.plot3D(points[:, 0], points[:, 1], points[:, 2])
        plt.show()
```

### Practice Problems

🔵 **Core Practice**

1. Vector Operations

   - Compute dot and cross products
   - Find vector projections
   - Solve distance problems
   - Analyze vector-valued functions

2. Motion Problems
   - Calculate velocity and acceleration
   - Find arc length
   - Determine curvature
   - Analyze particle motion

🟡 **Challenge Problems**

1. Advanced vector analysis
2. Frenet frame calculations
3. Complex motion problems
4. Vector field analysis

## Unit 2: Partial Derivatives (3 weeks)

### Learning Objectives

- Master partial differentiation techniques
- Understand gradient, divergence, and curl
- Apply chain rule to multivariable functions
- Solve optimization problems
- Implement numerical methods for partial derivatives

### Topics

🔵 **Foundations**

1. Functions of Several Variables

   - Domains and ranges
   - Level curves/surfaces
   - Limits and continuity
   - 3D visualization
   - Quadric surfaces

2. Partial Derivatives

   - Definition and notation
   - Higher-order partials
   - Mixed derivatives
   - Clairaut's theorem
   - Total differential
   - Linearization

3. Directional Derivatives
   - Definition
   - Gradient vector
   - Directional derivative formula
   - Normal vectors
   - Tangent planes
4. Chain Rule
   - Multivariable chain rule
   - Implicit differentiation
   - Related rates
   - Parametric surfaces

🟡 **Deeper Understanding**

1. Vector Calculus Concepts

   - Gradient fields
   - Conservative fields
   - Path independence
   - Potential functions
   - Divergence
   - Curl

2. Optimization
   - Critical points
   - Second derivatives test
   - Extrema on closed sets
   - Lagrange multipliers
   - Constrained optimization

🟢 **Theoretical Exploration**

1. Advanced Theory
   - Inverse function theorem
   - Implicit function theorem
   - Taylor's theorem
   - Mean value theorems

### Programming Implementations

```python
class PartialDerivatives:
    """Tools for computing and analyzing partial derivatives."""

    def __init__(self, func: Callable[[np.ndarray], float]):
        self.func = func

    def partial_derivative(self,
                         point: np.ndarray,
                         variable: int,
                         h: float = 1e-5) -> float:
        """
        Compute partial derivative with respect to specified variable.

        Args:
            point: Point at which to evaluate derivative
            variable: Index of variable for partial derivative
            h: Step size for numerical approximation
        """
        point_plus = point.copy()
        point_plus[variable] += h
        point_minus = point.copy()
        point_minus[variable] -= h

        return (self.func(point_plus) - self.func(point_minus)) / (2 * h)

    def gradient(self,
                point: np.ndarray,
                h: float = 1e-5) -> np.ndarray:
        """Compute gradient vector at given point."""
        return np.array([
            self.partial_derivative(point, i, h)
            for i in range(len(point))
        ])

    def directional_derivative(self,
                             point: np.ndarray,
                             direction: np.ndarray) -> float:
        """
        Compute directional derivative in given direction.
        Direction vector should be normalized.
        """
        grad = self.gradient(point)
        direction = direction / np.linalg.norm(direction)
        return np.dot(grad, direction)

    def hessian(self,
                point: np.ndarray,
                h: float = 1e-5) -> np.ndarray:
        """Compute Hessian matrix at given point."""
        n = len(point)
        H = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                point_pp = point.copy()
                point_pm = point.copy()
                point_mp = point.copy()
                point_mm = point.copy()

                point_pp[i] += h; point_pp[j] += h
                point_pm[i] += h; point_pm[j] -= h
                point_mp[i] -= h; point_mp[j] += h
                point_mm[i] -= h; point_mm[j] -= h

                H[i,j] = (self.func(point_pp) - self.func(point_pm) -
                         self.func(point_mp) + self.func(point_mm)) / (4 * h * h)

        return H
```

### Visualization Tools

```python
class MultivariableVisualizer:
    """Tools for visualizing multivariable functions."""

    def plot_surface(self, func: Callable[[float, float], float],
                    xlim: Tuple[float, float],
                    ylim: Tuple[float, float],
                    density: int = 50):
        """Plot 3D surface."""
        x = np.linspace(*xlim, density)
        y = np.linspace(*ylim, density)
        X, Y = np.meshgrid(x, y)
        Z = func(X, Y)

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, Z, cmap='viridis')
        fig.colorbar(surf)
        plt.show()

    def plot_contour(self, func: Callable[[float, float], float],
                    xlim: Tuple[float, float],
                    ylim: Tuple[float, float],
                    levels: int = 20):
        """Plot contour map."""
        x = np.linspace(*xlim, 100)
        y = np.linspace(*ylim, 100)
        X, Y = np.meshgrid(x, y)
        Z = func(X, Y)

        plt.figure(figsize=(10, 8))
        plt.contour(X, Y, Z, levels=levels)
        plt.colorbar()
        plt.show()
```

### Practice Problems

🔵 **Core Practice**

1. Basic Computations

   - Find partial derivatives
   - Calculate gradients
   - Evaluate directional derivatives
   - Find tangent planes

2. Optimization Problems
   - Find critical points
   - Classify extrema
   - Solve constrained optimization
   - Apply Lagrange multipliers

🟡 **Challenge Problems**

1. Vector field analysis
2. Complex optimization scenarios
3. Implicit function problems
4. Multi-constraint optimization

## Unit 3: Multiple Integration (4 weeks)

### Learning Objectives

- Master double and triple integrals
- Change variables using Jacobians
- Apply multiple integrals to physics problems
- Understand coordinate transformations
- Implement numerical integration methods

### Topics

🔵 **Foundations**

1. Double Integrals

   - Definition and properties
   - Iterated integrals
   - Fubini's theorem
   - Area and volume
   - Change of order
   - Average value

2. Double Integrals in Polar Coordinates

   - Coordinate transformation
   - Jacobian determinant
   - Area elements
   - Common applications
   - Integration strategy

3. Triple Integrals
   - Definition and setup
   - Integration order
   - Volume calculation
   - Mass and center of mass
   - Moment of inertia
4. Coordinate Systems
   - Cylindrical coordinates
   - Spherical coordinates
   - Change of variables
   - Integration techniques

🟡 **Deeper Understanding**

1. Advanced Applications

   - Fluid pressure
   - Mass distribution
   - Gravitational fields
   - Electric fields
   - Probability distributions

2. Change of Variables
   - General transformation
   - Multiple coordinate systems
   - Jacobian properties
   - Integration strategy

### Programming Implementations

```python
class MultipleIntegration:
    """Tools for numerical multiple integration."""

    def double_integral_rectangular(self,
                                  func: Callable[[float, float], float],
                                  xlim: Tuple[float, float],
                                  ylim: Tuple[float, float],
                                  nx: int = 100,
                                  ny: int = 100) -> float:
        """
        Compute double integral over rectangular region using
        two-dimensional midpoint rule.
        """
        x = np.linspace(xlim[0], xlim[1], nx)
        y = np.linspace(ylim[0], ylim[1], ny)
        dx = (xlim[1] - xlim[0]) / nx
        dy = (ylim[1] - ylim[0]) / ny

        X, Y = np.meshgrid(x, y)
        Z = func(X, Y)

        return np.sum(Z) * dx * dy

    def double_integral_polar(self,
                            func: Callable[[float, float], float],
                            r_lim: Tuple[float, float],
                            theta_lim: Tuple[float, float],
                            nr: int = 100,
                            ntheta: int = 100) -> float:
        """
        Compute double integral using polar coordinates.
        """
        r = np.linspace(r_lim[0], r_lim[1], nr)
        theta = np.linspace(theta_lim[0], theta_lim[1], ntheta)
        dr = (r_lim[1] - r_lim[0]) / nr
        dtheta = (theta_lim[1] - theta_lim[0]) / ntheta

        R, Theta = np.meshgrid(r, theta)
        X = R * np.cos(Theta)
        Y = R * np.sin(Theta)
        Z = func(X, Y) * R  # Include Jacobian factor

        return np.sum(Z) * dr * dtheta

    def triple_integral_rectangular(self,
                                  func: Callable[[float, float, float], float],
                                  limits: List[Tuple[float, float]],
                                  n: int = 50) -> float:
        """
        Compute triple integral over rectangular region.
        """
        x = np.linspace(limits[0][0], limits[0][1], n)
        y = np.linspace(limits[1][0], limits[1][1], n)
        z = np.linspace(limits[2][0], limits[2][1], n)
        dx = (limits[0][1] - limits[0][0]) / n
        dy = (limits[1][1] - limits[1][0]) / n
        dz = (limits[2][1] - limits[2][0]) / n

        X, Y, Z = np.meshgrid(x, y, z)
        V = func(X, Y, Z)

        return np.sum(V) * dx * dy * dz
```

### Visualization Helpers

```python
class IntegrationVisualizer:
    """Tools for visualizing multiple integrals."""

    def plot_double_integral_bounds(self,
                                  func: Callable[[float, float], float],
                                  region: Callable[[float], Tuple[float, float]],
                                  xlim: Tuple[float, float],
                                  density: int = 100):
        """
        Visualize region of integration and function surface.
        """
        x = np.linspace(xlim[0], xlim[1], density)
        y_bounds = np.array([region(xi) for xi in x])
        y_min, y_max = y_bounds.T

        # Plot region
        plt.figure(figsize=(12, 5))
        plt.subplot(121)
        plt.fill_between(x, y_min, y_max, alpha=0.3)
        plt.title('Region of Integration')

        # Plot surface
        ax = plt.subplot(122, projection='3d')
        X, Y = np.meshgrid(x, np.linspace(min(y_min), max(y_max), density))
        Z = func(X, Y)
        ax.plot_surface(X, Y, Z, cmap='viridis')
        plt.title('Function Surface')
        plt.show()
```

### Practice Problems

🔵 **Core Practice**

1. Double Integrals

   - Compute over rectangular regions
   - Change order of integration
   - Use polar coordinates
   - Find areas and volumes

2. Triple Integrals
   - Set up bounds
   - Choose coordinate system
   - Calculate physical quantities
   - Transform coordinates

🟡 **Challenge Problems**

1. Complex regions
2. Multiple coordinate systems
3. Physical applications
4. Optimization problems

## Unit 4: Vector Calculus (4 weeks)

### Learning Objectives

- Master vector field operations
- Understand line and surface integrals
- Apply fundamental theorems
- Analyze conservative and solenoidal fields
- Implement vector calculus computations

### Topics

🔵 **Foundations**

1. Vector Fields

   - Definition and visualization
   - Gradient fields
   - Flow lines
   - Conservative fields
   - Field operations
   - Potential functions

2. Line Integrals

   - Definition and properties
   - Work and circulation
   - Path independence
   - Green's theorem
   - Conservative field test
   - Fundamental theorem

3. Surface Integrals

   - Parametric surfaces
   - Surface area
   - Flux integrals
   - Orientation
   - Divergence theorem
   - Stokes' theorem

4. Fundamental Theorems
   - Green's theorem
   - Divergence theorem
   - Stokes' theorem
   - Applications
   - Connections

🟡 **Deeper Understanding**

1. Advanced Concepts

   - Differential forms
   - Manifolds
   - Exterior derivatives
   - General Stokes' theorem
   - Complex integration

2. Physical Applications
   - Electromagnetic theory
   - Fluid dynamics
   - Heat flow
   - Quantum mechanics
   - Elasticity theory

### Programming Implementations

```python
class VectorCalculus:
    """Tools for vector calculus computations and visualization."""

    def __init__(self):
        self.fig = plt.figure(figsize=(12, 8))

    def compute_curl(self,
                    F: Callable[[np.ndarray], np.ndarray],
                    point: np.ndarray,
                    h: float = 1e-5) -> np.ndarray:
        """
        Compute curl of vector field F at given point.
        F should take array of shape (3,) and return array of shape (3,).
        """
        curl = np.zeros(3)
        for i in range(3):
            j = (i + 1) % 3
            k = (i + 2) % 3

            # Compute partial derivatives
            point_p = point.copy()
            point_m = point.copy()
            point_p[j] += h
            point_m[j] -= h
            dFk_dj = (F(point_p)[k] - F(point_m)[k]) / (2 * h)

            point_p = point.copy()
            point_m = point.copy()
            point_p[k] += h
            point_m[k] -= h
            dFj_dk = (F(point_p)[j] - F(point_m)[j]) / (2 * h)

            curl[i] = dFk_dj - dFj_dk

        return curl

    def compute_divergence(self,
                          F: Callable[[np.ndarray], np.ndarray],
                          point: np.ndarray,
                          h: float = 1e-5) -> float:
        """Compute divergence of vector field F at given point."""
        div = 0
        for i in range(3):
            point_p = point.copy()
            point_m = point.copy()
            point_p[i] += h
            point_m[i] -= h
            div += (F(point_p)[i] - F(point_m)[i]) / (2 * h)
        return div

    def line_integral(self,
                     F: Callable[[np.ndarray], np.ndarray],
                     curve: Callable[[float], np.ndarray],
                     t_range: Tuple[float, float],
                     n: int = 1000) -> float:
        """
        Compute line integral of vector field F along curve.
        curve should be parameterized by t in t_range.
        """
        t = np.linspace(t_range[0], t_range[1], n)
        dt = (t_range[1] - t_range[0]) / (n - 1)

        # Compute curve points and tangent vectors
        points = np.array([curve(ti) for ti in t])
        tangents = np.array([curve(ti + dt) - curve(ti) for ti in t[:-1]])

        # Compute F·dr at each point
        F_values = np.array([F(p) for p in points[:-1]])
        products = np.sum(F_values * tangents, axis=1)

        return np.sum(products) * dt
```

### Visualization Tools

```python
class VectorFieldVisualizer:
    """Tools for visualizing vector fields and operations."""

    def plot_vector_field_2d(self,
                            F: Callable[[np.ndarray], np.ndarray],
                            xlim: Tuple[float, float],
                            ylim: Tuple[float, float],
                            density: int = 20):
        """Plot 2D vector field with streamlines."""
        x = np.linspace(*xlim, density)
        y = np.linspace(*ylim, density)
        X, Y = np.meshgrid(x, y)

        U = np.zeros_like(X)
        V = np.zeros_like(Y)
        for i in range(density):
            for j in range(density):
                point = np.array([X[i,j], Y[i,j]])
                field_value = F(point)
                U[i,j] = field_value[0]
                V[i,j] = field_value[1]

        plt.figure(figsize=(10, 8))
        plt.streamplot(X, Y, U, V, density=1.5, color='k', linewidth=1)
        plt.quiver(X, Y, U, V, alpha=0.3)
        plt.title('Vector Field with Streamlines')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()
```

### Practice Problems

🔵 **Core Practice**

1. Vector Fields

   - Compute curl and divergence
   - Find potential functions
   - Analyze field properties
   - Visualize vector fields

2. Line Integrals

   - Calculate work done
   - Test path independence
   - Find potential functions
   - Apply Green's theorem

3. Surface Integrals
   - Calculate flux
   - Apply divergence theorem
   - Use Stokes' theorem
   - Analyze field properties

🟡 **Challenge Problems**

1. Complex vector fields
2. Multiple surface integrals
3. Physical applications
4. Theoretical proofs

## Course Integration and Applications (2 weeks)

### Final Project Themes

🔵 **Core Integration Projects**

1. Physics Engine

```python
class PhysicsEngine:
    """3D physics simulation engine using multivariable calculus."""

    def __init__(self):
        self.vector_calc = VectorCalculus()
        self.integrator = MultipleIntegration()
        self.particles = []

    def add_particle(self,
                    mass: float,
                    position: np.ndarray,
                    velocity: np.ndarray):
        """Add particle to simulation."""
        self.particles.append({
            'mass': mass,
            'position': position,
            'velocity': velocity,
            'forces': []
        })

    def add_force_field(self,
                       field: Callable[[np.ndarray], np.ndarray]):
        """Add force field to simulation."""
        for particle in self.particles:
            particle['forces'].append(field)

    def simulate(self, dt: float, steps: int) -> List[np.ndarray]:
        """Run simulation for specified time steps."""
        trajectories = []
        for _ in range(steps):
            self._update(dt)
            trajectories.append([p['position'] for p in self.particles])
        return trajectories
```

2. Field Analysis Toolkit

```python
class FieldAnalyzer:
    """Comprehensive vector field analysis tools."""

    def analyze_field(self,
                     F: Callable[[np.ndarray], np.ndarray],
                     region: tuple) -> dict:
        """Complete analysis of vector field properties."""
        return {
            'conservative': self._test_conservative(F, region),
            'solenoidal': self._test_solenoidal(F, region),
            'potential': self._find_potential(F, region),
            'circulation': self._compute_circulation(F, region),
            'flux': self._compute_flux(F, region)
        }
```

### Advanced Applications

🔵 **Physical Systems**

1. Electromagnetic Fields

   - Electric field computation
   - Magnetic field analysis
   - Potential calculations
   - Field visualizations

2. Fluid Dynamics

   - Flow visualization
   - Pressure calculations
   - Circulation analysis
   - Streamline plotting

3. Gravitational Fields
   - Multi-body systems
   - Orbital mechanics
   - Field strength mapping
   - Potential energy analysis

🟡 **Mathematical Applications**

1. Optimization Problems

   - Constrained optimization
   - Surface fitting
   - Path optimization
   - Field optimization

2. Differential Geometry
   - Surface analysis
   - Curvature computation
   - Geodesic calculations
   - Minimal surfaces

### Transition to Advanced Topics

🔵 **Prerequisites for Advanced Courses**

1. Differential Geometry

   - Surface theory
   - Manifolds
   - Differential forms
   - Curvature concepts

2. Complex Analysis

   - Complex functions
   - Analytic functions
   - Complex integration
   - Residue theory

3. Tensor Analysis
   - Tensor algebra
   - Index notation
   - Coordinate transformations
   - Physical applications

### Resources and References

🔵 **Core Materials**

1. Textbooks

   - "Multivariable Calculus" by James Stewart
   - "Vector Calculus" by Marsden and Tromba
   - "Advanced Engineering Mathematics" by Kreyszig

2. Online Resources
   - MIT OpenCourseWare: Multivariable Calculus
   - 3Blue1Brown: Essence of Linear Algebra
   - Khan Academy: Multivariable Calculus

🟡 **Additional Materials**

1. Advanced Texts

   - "Differential Forms in Mathematical Physics" by Flanders
   - "Mathematical Methods in Physics" by Arfken
   - "Visual Complex Analysis" by Needham

2. Software Tools
   - Python Scientific Stack
   - Mathematica/Maple
   - GeoGebra 3D

### Course Completion Requirements

1. Core Competencies

   - Vector field analysis
   - Multiple integration
   - Coordinate transformations
   - Physical applications

2. Programming Skills

   - Numerical methods
   - Visualization
   - Implementation
   - Documentation

3. Project Portfolio
   - Physics simulation
   - Field analysis
   - Optimization problems
   - Visualization tools
