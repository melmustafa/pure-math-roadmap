# Single Variable Calculus

_Foundation Track A - Course 1_

## Course Overview

**Duration:** 3-4 months  
**Prerequisites:** High school algebra, basic trigonometry, functions  
**Programming:** Python basics  
**Time Commitment:** 10-15 hours/week

## Learning Path Structure

🔵 Core Content (Essential - Master These)  
🟡 Recommended (Deepen Understanding)  
🟢 Advanced (Extra Challenge)  
⭐ Applications (Real-world Usage)

## Unit 1: Limits and Continuity (3 weeks)

### Learning Objectives

- Understand and evaluate limits intuitively and algebraically
- Master continuity concepts and their implications
- Apply limit theorems to solve problems
- Implement basic numerical methods for limits

### Topics

🔵 **Foundations**

1. Limit Concept

   - Intuitive understanding
   - One-sided limits
   - Infinite limits
   - Limits at infinity
   - Limit laws

2. Computing Limits

   - Direct substitution
   - Factoring techniques
   - Rationalization
   - Trigonometric limits
   - Special limits (e.g., sinx/x)

3. Continuity
   - Definition of continuity
   - Types of discontinuities
   - Properties of continuous functions
   - Interval continuity

🟡 **Deeper Understanding**

1. ε-δ Definition

   - Formal limit definition
   - Proving limits exist
   - Sequential approach to limits

2. Theorems and Proofs
   - Intermediate Value Theorem
   - Extreme Value Theorem
   - Squeeze Theorem
   - Preservation theorems

🟢 **Theoretical Exploration**

1. Advanced Concepts
   - Uniform continuity
   - Darboux's theorem
   - Construction of nowhere continuous functions
   - Topology connections

⭐ **Applications**

1. Rate Problems

   - Instantaneous velocity
   - Population growth
   - Chemical reaction rates

2. Approximation
   - Numerical methods
   - Error estimation
   - Computer graphics applications

### Programming Implementations

```python
# Example implementations in the code/implementations/calculus/ directory
from typing import Callable, Optional
import numpy as np
import matplotlib.pyplot as plt

class LimitAnalysis:
    """Tools for analyzing and visualizing limits."""

    def __init__(self, func: Callable[[float], float]):
        self.func = func

    def evaluate_limit(self, x0: float, delta: float = 1e-6) -> tuple[float, float]:
        """Evaluate left and right limits at x0."""
        x_left = np.linspace(x0 - delta, x0, 1000)[:-1]
        x_right = np.linspace(x0, x0 + delta, 1000)[1:]

        y_left = self.func(x_left)
        y_right = self.func(x_right)

        return float(y_left[-1]), float(y_right[0])
```

### Practice Problems

🔵 **Core Practice**

1. Evaluate Basic Limits

   ```
   lim(x→2) (x² - 4)/(x - 2)
   lim(x→0) sin(x)/x
   lim(x→∞) (1 + 1/x)^x
   ```

2. Determine Continuity
   - Analyze piecewise functions
   - Find removable discontinuities
   - Apply IVT to prove solutions exist

🟡 **Challenge Problems**

1. Prove limit properties
2. Find ε-δ proofs
3. Analyze pathological functions

### Resources

🔵 **Primary Resources**

- Textbook: Stewart's Calculus, Chapters 1-2
- [MIT OCW: Single Variable Calculus](https://ocw.mit.edu/courses/mathematics/18-01sc-single-variable-calculus-fall-2010/)
- [3Blue1Brown: Essence of Calculus](https://www.youtube.com/playlist?list=PLZHQObOWTQDMsr9K-rj53DwVRMYO3t5Yr)

🟡 **Supplementary Materials**

- Spivak's Calculus (Theoretical)
- Paul's Online Math Notes
- Khan Academy Calculus

### Assessment

- Weekly problem sets
- Programming assignments
- Unit test covering:
  - Limit calculations
  - Continuity analysis
  - Theoretical understanding
  - Applications

## Unit 2: Differentiation (4 weeks)

### Learning Objectives

- Master the concept of derivatives as rates of change
- Apply differentiation rules effectively
- Understand the relationship between differentiability and continuity
- Solve applied optimization problems
- Implement numerical differentiation methods

### Topics

🔵 **Foundations**

1. Derivative Concept

   - Rate of change interpretation
   - Slope of tangent line
   - Notation (f'(x), dy/dx, d/dx)
   - One-sided derivatives
   - Differentiability implies continuity

2. Differentiation Rules

   - Power rule
   - Product rule
   - Quotient rule
   - Chain rule
   - Implicit differentiation
   - Logarithmic differentiation

3. Common Derivatives

   - Polynomial functions
   - Trigonometric functions
   - Exponential functions
   - Logarithmic functions
   - Inverse functions

4. Applications
   - Related rates
   - Linear approximation
   - Optimization problems
   - Newton's method
   - Error bounds

🟡 **Deeper Understanding**

1. Theoretical Foundations

   - Mean Value Theorem
   - Rolle's Theorem
   - L'Hôpital's Rule
   - Taylor's Theorem

2. Advanced Applications
   - Higher-order derivatives
   - Parametric differentiation
   - Marginal analysis
   - Elasticity (economics)
   - Motion analysis

🟢 **Theoretical Exploration**

1. Analysis Concepts
   - Darboux's Theorem
   - Derivative of nowhere differentiable functions
   - Weierstrass function
   - Connection to complex analysis

⭐ **Applications**

1. Physics

   - Velocity and acceleration
   - Force and motion
   - Simple harmonic motion
   - Wave propagation

2. Economics

   - Marginal cost/revenue
   - Profit optimization
   - Supply and demand
   - Growth models

3. Engineering
   - Rate of change problems
   - Optimization design
   - Control systems
   - Signal processing

### Programming Implementations

```python
class Differentiation:
    """Tools for numerical differentiation and analysis."""

    def __init__(self, func: Callable[[float], float]):
        self.func = func

    def forward_difference(self, x: float, h: float = 1e-5) -> float:
        """Compute derivative using forward difference."""
        return (self.func(x + h) - self.func(x)) / h

    def central_difference(self, x: float, h: float = 1e-5) -> float:
        """Compute derivative using central difference."""
        return (self.func(x + h) - self.func(x - h)) / (2 * h)

    def second_derivative(self, x: float, h: float = 1e-5) -> float:
        """Compute second derivative using central difference."""
        return (self.func(x + h) - 2 * self.func(x) + self.func(x - h)) / (h * h)

    def newton_method(self, x0: float, tol: float = 1e-6, max_iter: int = 100) -> float:
        """Find root using Newton's method."""
        x = x0
        for _ in range(max_iter):
            dx = self.central_difference(x)
            if abs(dx) < tol:
                break
            x = x - self.func(x) / dx
        return x
```

### Practice Problems

🔵 **Core Practice**

1. Basic Derivatives

   - Find derivatives of polynomial functions
   - Apply chain rule to composite functions
   - Use implicit differentiation
   - Solve related rates problems

2. Optimization Problems
   - Find maximum/minimum values
   - Solve word problems
   - Apply business optimization
   - Design optimization

🟡 **Challenge Problems**

1. Prove differentiation rules
2. Explore Mean Value Theorem applications
3. Analyze function behavior
4. Solve complex optimization problems

### Resources

🔵 **Primary Resources**

- Stewart's Calculus, Chapters 3-4
- [MIT OCW: Derivatives](https://ocw.mit.edu/courses/mathematics/18-01sc-single-variable-calculus-fall-2010/1.-differentiation/)
- [3Blue1Brown: Derivatives](https://www.youtube.com/watch?v=9vKqVkMQHKk)

🟡 **Supplementary Materials**

- Spivak's Calculus (Theoretical)
- [Paul's Online Notes: Derivatives](https://tutorial.math.lamar.edu/Classes/CalcI/CalcI.aspx)
- [Exercises in Differentiation](https://www.math.ubc.ca/~pwalls/math-python/differentiation/differentiation/)

### Assessment

1. Weekly Assignments

   - Problem sets covering all core topics
   - Programming implementations
   - Application problems

2. Unit Projects

   - Optimization case study
   - Numerical methods implementation
   - Real-world application analysis

3. Conceptual Understanding
   - Theory quizzes
   - Proof exercises
   - Concept mapping

## Unit 3: Integration (4 weeks)

### Learning Objectives

- Master the concept of definite and indefinite integrals
- Apply integration techniques effectively
- Understand the Fundamental Theorem of Calculus
- Solve applied problems using integration
- Implement numerical integration methods

### Topics

🔵 **Foundations**

1. Antiderivatives

   - Definition and notation
   - Basic antiderivative rules
   - Initial value problems
   - Indefinite integrals
   - Net change theorem

2. Definite Integrals

   - Riemann sums
   - Definition of definite integral
   - Properties of integrals
   - Average value of a function
   - Fundamental Theorem of Calculus

3. Integration Techniques

   - Basic integration rules
   - U-substitution
   - Integration by parts
   - Trigonometric integrals
   - Trigonometric substitution
   - Partial fractions
   - Integration tables

4. Applications
   - Area between curves
   - Volumes of revolution
   - Work calculations
   - Fluid pressure and force
   - Arc length
   - Surface area

🟡 **Deeper Understanding**

1. Advanced Integration

   - Improper integrals
   - Comparison tests
   - Parameter integrals
   - Double integrals preview
   - Integration strategies

2. Numerical Methods
   - Rectangle method
   - Trapezoidal rule
   - Simpson's rule
   - Error analysis
   - Adaptive methods

🟢 **Theoretical Exploration**

1. Analysis Concepts
   - Riemann vs. Lebesgue integration
   - Fundamental theorem proof
   - Integration in complex plane
   - Convergence theorems

⭐ **Applications**

1. Physics

   - Work and energy
   - Center of mass
   - Fluid pressure
   - Electric fields
   - Probability distributions

2. Engineering
   - Signal analysis
   - Beam deflection
   - Heat transfer
   - Control systems

### Programming Implementations

```python
class Integration:
    """Tools for numerical integration and analysis."""

    def __init__(self, func: Callable[[float], float]):
        self.func = func

    def riemann_sum(self, a: float, b: float, n: int, method: str = 'middle') -> float:
        """
        Compute Riemann sum using specified method.
        methods: 'left', 'right', 'middle'
        """
        dx = (b - a) / n
        x = np.linspace(a, b, n+1)
        if method == 'left':
            return dx * sum(self.func(x[:-1]))
        elif method == 'right':
            return dx * sum(self.func(x[1:]))
        else:  # middle
            return dx * sum(self.func((x[:-1] + x[1:]) / 2))

    def trapezoidal_rule(self, a: float, b: float, n: int) -> float:
        """Compute integral using trapezoidal rule."""
        dx = (b - a) / n
        x = np.linspace(a, b, n+1)
        y = self.func(x)
        return dx * (sum(y) - (y[0] + y[-1])/2)

    def simpsons_rule(self, a: float, b: float, n: int) -> float:
        """Compute integral using Simpson's rule."""
        if n % 2 != 0:
            n += 1  # ensure even number of intervals
        dx = (b - a) / n
        x = np.linspace(a, b, n+1)
        y = self.func(x)
        return dx/3 * (y[0] + y[-1] +
                      4*sum(y[1:-1:2]) +
                      2*sum(y[2:-1:2]))

    def adaptive_quadrature(self, a: float, b: float, tol: float = 1e-6) -> float:
        """Adaptive quadrature method for automatic error control."""
        pass  # Implementation details
```

### Practice Problems

🔵 **Core Practice**

1. Basic Integration

   - Find antiderivatives
   - Evaluate definite integrals
   - Apply u-substitution
   - Use integration by parts

2. Applied Problems
   - Calculate areas and volumes
   - Solve work problems
   - Find fluid force
   - Determine arc length

🟡 **Challenge Problems**

1. Complex integration techniques
2. Improper integral evaluation
3. Numerical method implementation
4. Real-world modeling problems

### Resources

🔵 **Primary Resources**

- Stewart's Calculus, Chapters 5-7
- [MIT OCW: Integration](https://ocw.mit.edu/courses/mathematics/18-01sc-single-variable-calculus-fall-2010/unit-3-integration/)
- [3Blue1Brown: Integration](https://www.youtube.com/watch?v=rfG8ce4nNh0)

🟡 **Supplementary Materials**

- [Integration Techniques](https://tutorial.math.lamar.edu/Classes/CalcI/IntegrationIntro.aspx)
- [Numerical Integration Methods](https://www.math.ubc.ca/~pwalls/math-python/integration/integration/)
- Interactive Demonstrations

### Assessment

1. Weekly Problem Sets

   - Integration techniques
   - Applications
   - Numerical methods

2. Programming Projects

   - Implement numerical integration
   - Compare method accuracy
   - Solve applied problems

3. Conceptual Understanding
   - Method selection
   - Error analysis
   - Theoretical foundations

## Unit 4: Sequences and Series (3 weeks)

### Learning Objectives

- Understand sequences and their convergence
- Master series convergence tests
- Work with power series
- Apply Taylor series expansions
- Implement series computations

### Topics

🔵 **Foundations**

1. Sequences

   - Definition and notation
   - Convergence and divergence
   - Bounded sequences
   - Monotonic sequences
   - Limit properties of sequences
   - Common sequences

2. Series Basics

   - Series definition
   - Partial sums
   - Geometric series
   - Telescoping series
   - p-series
   - Harmonic series

3. Convergence Tests

   - nth term test
   - Integral test
   - Comparison tests
   - Limit comparison test
   - Ratio test
   - Root test
   - Alternating series test

4. Power Series
   - Interval of convergence
   - Radius of convergence
   - Term-by-term differentiation
   - Term-by-term integration
   - Common power series

🟡 **Deeper Understanding**

1. Advanced Series

   - Absolute convergence
   - Conditional convergence
   - Rearrangements
   - Double series
   - Product series

2. Taylor Series
   - Taylor's theorem
   - Maclaurin series
   - Error bounds
   - Common expansions
   - Applications

🟢 **Theoretical Exploration**

1. Analysis Concepts
   - Completeness of R
   - Cauchy sequences
   - Uniform convergence
   - Weierstrass M-test
   - Abel's theorem

⭐ **Applications**

1. Applied Series
   - Function approximation
   - Numerical computation
   - Signal processing
   - Physics applications
   - Error analysis

### Programming Implementations

```python
class SeriesAnalysis:
    """Tools for analyzing sequences and series."""

    def sequence_limit(self, seq_func: Callable[[int], float],
                      n_terms: int = 100,
                      tol: float = 1e-6) -> Optional[float]:
        """Estimate the limit of a sequence."""
        terms = [seq_func(n) for n in range(1, n_terms + 1)]
        if len(terms) < 2:
            return None

        # Check for convergence
        for i in range(len(terms) - 1, 0, -1):
            if abs(terms[i] - terms[i-1]) < tol:
                return terms[i]
        return None

    def geometric_series_sum(self, a: float, r: float, n: Optional[int] = None) -> float:
        """
        Compute sum of geometric series.
        If n is None, compute sum of infinite series (|r| < 1 only).
        """
        if n is None:
            if abs(r) >= 1:
                raise ValueError("Infinite series requires |r| < 1")
            return a / (1 - r)
        return a * (1 - r**n) / (1 - r)

    def ratio_test(self, series_terms: List[float], tol: float = 1e-6) -> dict:
        """Apply ratio test to determine series convergence."""
        ratios = [abs(series_terms[i+1] / series_terms[i])
                 for i in range(len(series_terms)-1)]
        limit = self.sequence_limit(lambda n: ratios[n-1], len(ratios))

        return {
            'limit': limit,
            'converges': limit is not None and limit < 1,
            'diverges': limit is not None and limit > 1,
            'inconclusive': limit is None or abs(limit - 1) < tol
        }

    def taylor_series(self, func: Callable, x0: float, n: int) -> np.ndarray:
        """
        Generate coefficients for Taylor series expansion.
        Uses numerical differentiation for higher derivatives.
        """
        coefficients = []
        for i in range(n):
            # Implement numerical derivatives
            pass
        return np.array(coefficients)
```

### Practice Problems

🔵 **Core Practice**

1. Sequence Problems

   - Find sequence limits
   - Prove convergence/divergence
   - Find recursive formulas
   - Analyze monotonicity

2. Series Problems
   - Test for convergence
   - Find sums of series
   - Determine intervals of convergence
   - Work with power series

🟡 **Challenge Problems**

1. Advanced convergence analysis
2. Taylor series applications
3. Error bound calculations
4. Series transformations

### Resources

🔵 **Primary Resources**

- Stewart's Calculus, Chapters 11-12
- [MIT OCW: Series](https://ocw.mit.edu/courses/mathematics/18-01sc-single-variable-calculus-fall-2010/unit-5-exploring-the-infinite/)
- [3Blue1Brown: Power Series](https://www.youtube.com/watch?v=3d6DsjIBzJ4)

🟡 **Supplementary Materials**

- [Series and Sequences](https://tutorial.math.lamar.edu/Classes/CalcII/SeriesIntro.aspx)
- [Power Series Applications](https://www.math.ubc.ca/~pwalls/math-python/series/power-series/)
- Interactive Series Visualizations

### Assessment

1. Weekly Assignments

   - Convergence tests
   - Series calculations
   - Taylor series problems

2. Programming Projects

   - Series calculator implementation
   - Convergence test automation
   - Taylor series visualization

3. Conceptual Understanding
   - Test selection strategy
   - Error analysis
   - Applications

## Unit 5: Course Integration and Applications (2 weeks)

### Integration Project Themes

🔵 **Core Projects**

1. Numerical Methods Package

   ```python
   class CalculusToolkit:
       """Comprehensive calculus computational toolkit."""

       def __init__(self):
           self.differentiation = DifferentiationMethods()
           self.integration = IntegrationMethods()
           self.series = SeriesAnalysis()

       def analyze_function(self, func: Callable,
                          interval: tuple[float, float]) -> dict:
           """Complete analysis of a function."""
           return {
               'derivatives': self._analyze_derivatives(func, interval),
               'integrals': self._analyze_integrals(func, interval),
               'series': self._analyze_series(func, interval)
           }
   ```

2. Real-world Applications

   - Physics simulations
   - Economic modeling
   - Engineering problems
   - Data analysis

3. Visualization Tools
   - Function explorer
   - Derivative visualizer
   - Integration animator
   - Series convergence plots

🟡 **Extended Projects**

1. Advanced Topics Integration

   - Differential equations preview
   - Vector calculus introduction
   - Complex analysis preview

2. Mathematical Modeling
   - Population dynamics
   - Financial mathematics
   - Physical systems
   - Optimization problems

## Final Review and Resources

### Comprehensive Review

🔵 **Core Topics Review**

1. Limits and Continuity

   - Key concepts
   - Problem-solving strategies
   - Common pitfalls
   - Applications

2. Derivatives

   - Differentiation rules
   - Applications
   - Optimization
   - Related rates

3. Integration

   - Integration techniques
   - Applications
   - Numerical methods
   - Problem-solving strategies

4. Series
   - Convergence tests
   - Power series
   - Taylor series
   - Applications

### Study Resources

🔵 **Essential Resources**

1. Textbooks

   - Primary: Stewart's Calculus
   - Theory: Spivak's Calculus
   - Problems: MIT OCW Problem Sets

2. Online Resources

   - [MIT OpenCourseWare](https://ocw.mit.edu/courses/mathematics/18-01sc-single-variable-calculus-fall-2010/)
   - [3Blue1Brown Series](https://www.3blue1brown.com/topics/calculus)
   - [Paul's Online Notes](https://tutorial.math.lamar.edu/)

3. Interactive Tools
   - Wolfram Alpha
   - Desmos
   - GeoGebra
   - Custom Python implementations

## Future Directions

### Next Steps

🔵 **Preparation for Advanced Courses**

1. Multivariable Calculus

   - Vector functions
   - Partial derivatives
   - Multiple integrals
   - Vector calculus

2. Differential Equations

   - First-order equations
   - Linear equations
   - Systems of equations
   - Applications

3. Analysis
   - Real analysis
   - Theoretical foundations
   - Rigorous proofs
   - Advanced concepts

### Advanced Topics Preview

🟡 **Extended Mathematics**

1. Vector Calculus

   - Vector fields
   - Line integrals
   - Surface integrals
   - Fundamental theorems

2. Complex Analysis
   - Complex functions
   - Analytic functions
   - Complex integration
   - Series expansions

## Course Completion Checklist

### Required Mastery

- [ ] Limit evaluation and properties
- [ ] Derivative computation and applications
- [ ] Integration techniques and applications
- [ ] Series convergence and applications
- [ ] Basic numerical methods
- [ ] Problem-solving strategies
- [ ] Programming implementations

### Recommended Additional Skills

- [ ] Advanced proof techniques
- [ ] Numerical analysis understanding
- [ ] Mathematical modeling
- [ ] Programming proficiency
- [ ] Visualization creation

## Final Notes

### Success Strategies

1. Regular practice with problems
2. Implementation of concepts in code
3. Focus on understanding over memorization
4. Connect theories with applications
5. Build on fundamentals before advancing

### Common Pitfalls to Avoid

1. Memorizing without understanding
2. Skipping foundational concepts
3. Insufficient practice
4. Avoiding difficult problems
5. Neglecting programming practice
