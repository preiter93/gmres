//! Gmres iteration
//!
//! Generalized minimum residual method to iteratively solve
//! `A x = b`.
//!
//! First draft for `ndarray_linalg`
//!
//! # Documentation
//!
//! <a href="https://github.com/preiter93/gmres/blob/master/doc/gmres.pdf" target="_blank">Algorithm</a>
//!
//! # TODO
//! - Preconditioner : Done
//! - Error Handling
//! - Test complex values: Done
//! - Use relative residual
use ndarray::{azip, s, Array1, Array2, ArrayBase, DataMut, Ix1};
use ndarray_linalg::{
    error::Result, krylov::AppendResult, krylov::Orthogonalizer, krylov::MGS, norm::Norm,
    operator::LinearOperator, types::Scalar, Lapack, SolveTriangular,
};
use ndarray_linalg::{Diag, UPLO};
use num_traits::One;
use std::iter::Iterator;

/// X-vector
type X<A> = Array1<A>;

/// Residual
type R<A> = <A as Scalar>::Real;

/// Gmres Status
#[derive(Clone, Debug)]
pub enum GmresStatus {
    /// Gmres converged
    Converged,
    /// Gmres not converged
    NotConverged,
    /// Krylov vectors are linearly dependent
    KrylovDependent,
}

// Gmres iterator
pub struct Gmres<'a, A, S, F, Ortho>
where
    A: Scalar,
    S: DataMut<Elem = A>,
    F: LinearOperator<Elem = A>,
    Ortho: Orthogonalizer<Elem = A>,
{
    /// Linear operator
    a: &'a F,
    /// Initial guess
    x0: ArrayBase<S, Ix1>,
    /// Next vector (normalized `|v|=1`)
    v: Array1<A>,
    /// Orthogonalizer
    ortho: Ortho,
    /// Current iteration number
    m: usize,
    /// Maximum number of iterations
    maxiter: usize,
    /// Tolerance for Gmres convergence
    tol: A::Real,
    /// `r` = Givens_rotation(H)
    r: Vec<Array1<A>>,
    /// `g` = Givens_rotation(`|r0|e1`)
    g: Vec<A>,
    /// Cosine component of Givens matrix
    cs: Vec<A>,
    /// Sine component of Givens matrix
    sn: Vec<A>,
    /// Residual
    residual: A::Real,
    /// Preconditioner (optional)
    pc: Option<F>,
    /// Status
    status: GmresStatus,
}

impl<'a, A, S, F, Ortho> Gmres<'a, A, S, F, Ortho>
where
    A: Scalar + Lapack,
    S: DataMut<Elem = A>,
    F: LinearOperator<Elem = A> + 'a,
    Ortho: Orthogonalizer<Elem = A>,
{
    /// Create a Gmres iterator from any linear operator `a`
    ///
    /// # Panics
    /// - `maxiter` > `b.len()`
    #[allow(clippy::many_single_char_names)]
    pub fn new(a: &'a F, b: &ArrayBase<S, Ix1>, x0: ArrayBase<S, Ix1>, mut ortho: Ortho) -> Self {
        assert_eq!(ortho.len(), 0);
        assert!(ortho.tolerance() < One::one());
        // First Krylov vector
        let mut v = b - a.apply(&x0);
        // normalize before append
        let norm = v.norm_l2();
        azip!((v in &mut v)  *v = v.div_real(norm));
        ortho.append(v.view());
        let tol = <A>::real(1e-8_f32);
        let residual = norm;
        let status = if residual <= tol {
            GmresStatus::Converged
        } else {
            GmresStatus::NotConverged
        };

        Gmres {
            a,
            x0,
            v,
            ortho,
            m: 0,
            maxiter: b.len(),
            tol,
            r: vec![],
            g: vec![A::from(norm).unwrap()],
            cs: vec![],
            sn: vec![],
            residual,
            pc: None,
            status,
        }
    }

    /// Set Maximum number of iterations
    #[must_use]
    pub fn maxiter(mut self, maxiter: usize) -> Self {
        self.maxiter = maxiter;
        self
    }

    /// Set convergence tolerance
    #[must_use]
    pub fn tolerance(mut self, tol: A::Real) -> Self {
        // Update status
        self.status = match self.status {
            GmresStatus::Converged | GmresStatus::NotConverged => {
                if self.residual <= tol {
                    GmresStatus::Converged
                } else {
                    GmresStatus::NotConverged
                }
            }
            GmresStatus::KrylovDependent => GmresStatus::KrylovDependent,
        };
        self.tol = tol;
        self
    }

    /// Set preconditioner
    #[must_use]
    pub fn preconditioner(mut self, pc: F) -> Self {
        self.pc = Some(pc);
        self
    }

    /// Dimension of problem
    pub fn dim(&self) -> usize {
        self.x0.len()
    }

    /// Dimension of Krylov subspace
    pub fn dim_krylov(&self) -> usize {
        self.ortho.len()
    }

    /// Return residual
    pub fn residual(&self) -> R<A> {
        self.residual
    }

    /// Return current iteration numer
    pub fn iteration(&self) -> usize {
        self.m
    }

    /// Return status
    pub fn status(&self) -> GmresStatus {
        self.status.clone()
    }

    /// Calculate the givens rotation
    /// [ cs        sn] [ f ]   [ r ]
    /// [-sn        cs] [ g ] = [ 0 ]
    ///
    /// # Parameters
    /// `f`: Scalar
    /// `g`: Scalar
    ///
    /// # Returns
    /// `cs`: The cosine of the rotation
    /// `sn`: The sine of the rotation
    fn giv_rot(f: A, g: A) -> (A, A) {
        let t = (f * f + g * g).sqrt();
        (f / t, g / t)
    }

    /// Apply givens rotation to h
    ///
    /// `hnew = J_k J_(k-1) .. J_1 h`
    ///
    /// where `J_k` is the k-th givens rotation matrix.
    /// Its components are provided through `cs` and `sn`.
    ///
    /// hnew is zero on its last entry.
    ///
    /// # Parameters
    /// `h`   : vector of size k + 1
    /// `cs`  : vector of size k
    /// `sn`  : vector of size k

    /// Returns:
    /// `hnew`: updated h, mutates h inplace
    /// `cs_k`: cos component of k+1-th Givens matrix
    /// `sn_k`: sin component of k+1-th Givens matrix
    fn apply_giv_rot<S1: DataMut<Elem = A>>(
        h: &mut ArrayBase<S1, Ix1>,
        cs: &[A],
        sn: &[A],
    ) -> (A, A) {
        assert!(cs.len() == sn.len());
        assert!(cs.len() == h.len() - 2);
        let k = cs.len();
        // Apply for i-th column
        for i in 0..k {
            let tmp = cs[i] * h[i] + sn[i] * h[i + 1];
            h[i + 1] = -sn[i] * h[i] + cs[i] * h[i + 1];
            h[i] = tmp;
        }
        // Update the next sin / cos values for Givens rotation
        let (cs_k, sn_k) = Self::giv_rot(h[k], h[k + 1]);
        // Eliminate h[k+1]
        h[k] = cs_k * h[k] + sn_k * h[k + 1];
        h[k + 1] = A::zero();
        (cs_k, sn_k)
    }

    /// Iterate until convergent
    ///
    /// Returns result and vector of residuals
    ///
    /// # Errors
    /// - `solve_triangular` fails
    pub fn complete(mut self) -> Result<(X<A>, Vec<R<A>>)> {
        // Iterate until completion and collect residual
        let mut residuals = vec![self.residual];
        for r in &mut self {
            residuals.push(r);
        }
        let x = self.finalize()?;
        Ok((x, residuals))
    }

    /// Finalize iterator, return solution and residual
    ///
    /// # Errors
    /// - `solve_triangular` fails
    pub fn finalize(mut self) -> Result<X<A>> {
        // min |g ??? R y| for y, where R is upper triangular
        let mut r: Array2<A> = Array2::zeros((self.m, self.m));
        for (j, col) in self.r.iter().enumerate() {
            for (i, v) in col.iter().enumerate() {
                r[[i, j]] = *v;
            }
        }
        self.g.pop();
        let y = r.solve_triangular(
            UPLO::Upper,
            Diag::NonUnit,
            &Array1::from_vec(self.g.clone()),
        )?;
        // Update x = x0 + z
        let mut z = self.ortho.get_q().dot(&y);
        if let Some(pc) = &self.pc {
            pc.apply_mut(&mut z);
        }
        let x = &self.x0 + &z;
        Ok(x)
    }
}

impl<'a, A, S, F, Ortho> Iterator for Gmres<'a, A, S, F, Ortho>
where
    A: Scalar + Lapack,
    S: DataMut<Elem = A>,
    F: LinearOperator<Elem = A>,
    Ortho: Orthogonalizer<Elem = A>,
{
    type Item = A::Real;

    fn next(&mut self) -> Option<Self::Item> {
        match self.status {
            GmresStatus::Converged | GmresStatus::KrylovDependent => return None,
            GmresStatus::NotConverged => (),
        };
        // Maximum number of iterations reached
        if self.m >= self.maxiter {
            return None;
        }

        // Number of current iteration
        let j = self.m;

        // (1) Generate new Krylov vector
        if let Some(pc) = &self.pc {
            pc.apply_mut(&mut self.v);
        }
        self.a.apply_mut(&mut self.v);
        let result = self.ortho.div_append(&mut self.v);
        let norm = self.v.norm_l2();
        azip!((v in &mut self.v) *v = v.div_real(norm));
        // If dependent, it is catched in next iteration
        let mut h = match result {
            AppendResult::Added(coef) => coef,
            AppendResult::Dependent(coef) => {
                self.status = GmresStatus::KrylovDependent;
                coef
            }
        };

        // (2) Apply Givens rotation
        let (cs_k, sn_k) = Self::apply_giv_rot(&mut h, &self.cs, &self.sn);
        self.cs.push(cs_k);
        self.sn.push(sn_k);
        self.r.push(h.slice(s![..h.len() - 1]).to_owned());
        self.g.push(-self.sn[j] * self.g[j]);
        self.g[j] = self.cs[j] * self.g[j];

        // (3) Check residual
        self.residual = self.g[self.g.len() - 1].abs();
        if self.residual <= self.tol {
            self.status = GmresStatus::Converged;
        };
        self.m += 1;
        Some(self.residual)
    }
}

/// Generalized minimum residual method to iteratively solve
///     `A x = b`.
/// using modified Gram-Schmidt orthogonalizer
///
/// # Parameters
/// `a`           : Linear Matrix Operator
/// `b`           : Array1, right-hand-side
/// `x0`          : Array1, initial guess (optional)
/// `maxiter`     : Maximum number of Gmres iterations
/// `tol_mgs`     : Convergence tolerance for Gram-Schmidth orthogonalizer
/// `tol_gmres`   : Convergence tolerance for Gmres residual
///
/// # Errors
/// - `solve_triangular` fails
pub fn gmres_mgs<'a, A, S, F>(
    a: &'a F,
    b: &ArrayBase<S, Ix1>,
    x0: ArrayBase<S, Ix1>,
    maxiter: usize,
    tol_mgs: A::Real,
    tol_gmres: A::Real,
) -> Result<(X<A>, Vec<R<A>>)>
where
    A: Scalar + Lapack,
    S: DataMut<Elem = A>,
    F: LinearOperator<Elem = A> + 'a,
{
    let mgs = MGS::new(b.len(), tol_mgs);
    Gmres::new(a, b, x0, mgs)
        .maxiter(maxiter)
        .tolerance(tol_gmres)
        .complete()
}

#[cfg(test)]
mod test {
    //! cargo test -- --show-output
    use super::*;
    use ndarray::Zip;
    use ndarray::{Data, Dimension};
    use ndarray_linalg::types::c64;
    use ndarray_rand::rand_distr::Uniform;
    use ndarray_rand::RandomExt;

    /// Test approx equality of two arrays element-wise
    ///
    /// # Panics
    /// Panics when difference is larger than 1e-3.
    pub fn approx_eq<A, S, D>(x: &ArrayBase<S, D>, y: &ArrayBase<S, D>)
    where
        A: Scalar,
        S: Data<Elem = A>,
        D: Dimension,
    {
        let tol = <A>::real(1e-5_f32);
        for (a, b) in x.iter().zip(y.iter()) {
            assert!(((*a - *b).abs() < tol), "Got {} vs {}.", b, a);
        }
    }

    #[test]
    /// Variable size, random input
    fn test_real() {
        let m = 28;
        let a = Array2::<f64>::random((m, m), Uniform::new(0., 10.));
        let b = Array1::<f64>::random(m, Uniform::new(0., 10.));
        let x0: Array1<f64> = Array1::zeros(b.len());
        let maxiter = b.len();
        let (x, e) = gmres_mgs(&a, &b, x0, maxiter, 1e-8, 1e-8).unwrap();
        let (size, iter, residual) = (b.len(), e.len() - 1, e[e.len() - 1]);
        println!("Size {:?} | iter {:?} | Res {:4.2e}", size, iter, residual);
        approx_eq(&b, &a.dot(&x));
    }

    #[test]
    /// Test with complex arrays
    fn test_gmres_complex() {
        let m = 28;
        let a = array2_rand_complex(m, m);
        let b = array1_rand_complex(m);
        let x0: Array1<c64> = Array1::zeros(b.len());
        let maxiter = b.len();
        let (x, e) = gmres_mgs(&a, &b, x0, maxiter, 1e-8, 1e-8).unwrap();
        let (size, iter, residual) = (b.len(), e.len() - 1, e[e.len() - 1]);
        println!("Size {:?} | iter {:?} | Res {:4.2e}", size, iter, residual);
        approx_eq(&b, &a.dot(&x));
    }

    fn array2_rand_complex(m: usize, n: usize) -> Array2<c64> {
        let re = Array2::<f64>::random((m, n), Uniform::new(-1.0, 1.0));
        let im = Array2::<f64>::random((m, n), Uniform::new(-1.0, 1.0));
        let mut c = Array2::<c64>::zeros((m, n));
        Zip::from(&mut c).and(&re).and(&im).for_each(|c, &re, &im| {
            *c = c64 { re, im };
        });
        c
    }

    fn array1_rand_complex(n: usize) -> Array1<c64> {
        let re = Array1::<f64>::random(n, Uniform::new(-1.0, 1.0));
        let im = Array1::<f64>::random(n, Uniform::new(-1.0, 1.0));
        let mut c = Array1::<c64>::zeros(n);
        Zip::from(&mut c).and(&re).and(&im).for_each(|c, &re, &im| {
            *c = c64 { re, im };
        });
        c
    }
}

// #[test]
// /// Fixed size, fixed input
// fn test1() {
//     let a = array![[1., 2., 3.], [3., 4., 5.], [4., 7., 8.]];
//     let b = array![3., 2., 7.];
//     let x0: Array1<f64> = Array1::zeros(b.len());
//     let maxiter = b.len();
//     let (x, e) = gmres_mgs(&a, &b, x0, maxiter, 1e-8, 1e-8).unwrap();
//     let (size, iter, residual) = (b.len(), e.len() - 1, e[e.len() - 1]);
//     println!("Size {:?} | iter {:?} | Res {:4.2e}", size, iter, residual);
//     approx_eq(&b, &a.dot(&x));
// }

// /// Finalize iterator, return solution and residual
// pub fn finalize(mut self) -> GmresResult<A> {
//     // min |g ??? R y| for y, where R is upper triangular
//     let mut r: Array2<A> = Array2::zeros((self.m, self.m));
//     for (j, col) in self.r.iter().enumerate() {
//         for (i, v) in col.iter().enumerate() {
//             r[[i, j]] = *v;
//         }
//     }
//     self.g.pop();
//     match r.solve_triangular(
//         UPLO::Upper,
//         Diag::NonUnit,
//         &Array1::from_vec(self.g.clone()),
//     ) {
//         // Update x = x0 + z
//         Ok(y) => {
//             let mut z = self.ortho.get_q().dot(&y);
//             if let Some(pc) = &self.pc {
//                 pc.apply_mut(&mut z);
//             }
//             let x = &self.x0 + &z;
//             let residual = self.residual();
//             if residual[residual.len() - 1] <= self.tol {
//                 GmresResult::Converged((x, residual))
//             } else {
//                 GmresResult::NotConverged((x, residual))
//             }
//         }
//         Err(_) => GmresResult::Error,
//     }
// }
