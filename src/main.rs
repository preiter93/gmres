//! Gmres iteration
//!
//! Generalized minimum residual method to iteratively solve
//! `A x = b`.
//!
//! First draft for `ndarray_linalg`
use ndarray::{azip, s, Array1, Array2, ArrayBase, DataMut, Ix1};
use ndarray_linalg::{
    krylov::AppendResult, krylov::Orthogonalizer, krylov::MGS, norm::Norm,
    operator::LinearOperator, types::Scalar, Lapack, SolveTriangular,
};
use ndarray_linalg::{Diag, UPLO};
use num_traits::One;
use std::iter::Iterator;

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
    /// `r` = Givens_rotation(H)
    r: Vec<Array1<A>>,
    /// `g` = Givens_rotation(`|r0|e1`)
    g: Vec<A>,
    /// Cosine component of Givens matrix
    cs: Vec<A>,
    /// Sine component of Givens matrix
    sn: Vec<A>,
    /// Residual
    e: Vec<<A as Scalar>::Real>,
    // /// Right hand side
    // b: ArrayBase<S, Ix1>,
    // /// Arnoldi iterator
    // arnoldi: Arnoldi<A, OwnedRepr<A>, &'a F, Ortho>
    // /// Tolerance for convergence
    // tol: <A as Scalar>::Real,
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
    pub fn new(
        a: &'a F,
        b: &ArrayBase<S, Ix1>,
        x0: ArrayBase<S, Ix1>,
        mut ortho: Ortho,
        maxiter: usize,
        // tol: <A as Scalar>::Real,
    ) -> Self {
        assert_eq!(ortho.len(), 0);
        assert!(ortho.tolerance() < One::one());
        assert!(maxiter <= b.len());
        // First Krylov vector
        let mut v = b - a.apply(&x0);
        // normalize before append
        let norm = v.norm_l2();
        azip!((v in &mut v)  *v = v.div_real(norm));
        ortho.append(v.view());
        // Additional storage for Givens rotation
        let r = vec![];
        let g = vec![A::from(norm).unwrap()];
        let e = vec![norm];
        let cs = vec![];
        let sn = vec![];
        let m = 0;

        Gmres {
            a,
            x0,
            v,
            ortho,
            m,
            maxiter,
            r,
            g,
            cs,
            sn,
            e,
        }
    }

    /// Dimension of Krylov subspace
    pub fn dim(&self) -> usize {
        self.ortho.len()
    }

    /// Return residual
    pub fn residual(&self) -> Vec<A::Real> {
        self.e.clone()
    }

    /// Calculate the givens rotation
    /// [ cs        sn] [ f ]   [ r ]
    /// [-conj(sn)  cs] [ g ] = [ 0 ]
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
    /// # Panics
    /// - Fail of triangular solve
    pub fn complete(mut self, tol: <A as Scalar>::Real) -> (Array1<A>, Vec<A::Real>) {
        for err in &mut self {
            if err <= tol {
                break;
            }
        }
        // min |g − R y| for y, where R is upper triangular
        let mut r: Array2<A> = Array2::zeros((self.m, self.m));
        for (j, col) in self.r.iter().enumerate() {
            for (i, v) in col.iter().enumerate() {
                r[[i, j]] = *v;
            }
        }
        let diag = Diag::NonUnit;
        let uplo = UPLO::Upper;
        self.g.pop();
        let g = Array1::from_vec(self.g.clone());
        // TODO: Handle error
        let y: Array1<A> = r.solve_triangular(uplo, diag, &g).unwrap();
        // Update x = x0 + Q y
        let x = &self.x0 + &self.ortho.get_q().dot(&y);
        let residual = self.residual();
        (x, residual)
    }
}

impl<'a, A, S, F, Ortho> Iterator for Gmres<'a, A, S, F, Ortho>
where
    A: Scalar + Lapack,
    S: DataMut<Elem = A>,
    F: LinearOperator<Elem = A>,
    Ortho: Orthogonalizer<Elem = A>,
{
    type Item = <A as Scalar>::Real;

    fn next(&mut self) -> Option<Self::Item> {
        // Maximum number of iterations reached
        if self.m >= self.maxiter {
            return None;
        }
        let j = self.m;
        // (1) Generate new Krylov vector
        self.a.apply_mut(&mut self.v);
        let result = self.ortho.div_append(&mut self.v);
        let norm = self.v.norm_l2();
        azip!((v in &mut self.v) *v = v.div_real(norm));
        // TODO: Break Gmres if Dependent
        let mut h = match result {
            AppendResult::Added(coef) => coef,
            AppendResult::Dependent(coef) => coef,
        };

        // (2) Apply Givens rotation
        let (cs_k, sn_k) = Self::apply_giv_rot(&mut h, &self.cs, &self.sn);
        self.cs.push(cs_k);
        self.sn.push(sn_k);
        self.r.push(h.slice(s![..h.len() - 1]).to_owned());
        self.g.push(-self.sn[j] * self.g[j]);
        self.g[j] = self.cs[j] * self.g[j];

        // (3) Check residual
        // TODO: Is abs() ok?
        let err = self.g[self.g.len() - 1].abs();
        self.e.push(err);
        self.m += 1;
        Some(err)
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
pub fn gmres_mgs<'a, A, S, F>(
    a: &'a F,
    b: &ArrayBase<S, Ix1>,
    x0: ArrayBase<S, Ix1>,
    maxiter: usize,
    tol_mgs: A::Real,
    tol_gmres: A::Real,
) -> (Array1<A>, Vec<A::Real>)
where
    A: Scalar + Lapack,
    S: DataMut<Elem = A>,
    F: LinearOperator<Elem = A> + 'a,
{
    let mgs = MGS::new(b.len(), tol_mgs);
    Gmres::new(a, b, x0, mgs, maxiter).complete(tol_gmres)
}

fn main() {
    use ndarray::array;
    println!("Hello, world!");
    let a = array![[1., 2., 3.], [3., 4., 5.], [4., 7., 8.]];
    let b = array![3., 2., 7.];
    let x0: Array1<f64> = Array1::zeros(b.len());
    let maxiter = b.len();
    let tol_mgs = 1e-8;
    let tol_gmres = 1e-8;
    let (x, res) = gmres_mgs(&a, &b, x0, maxiter, tol_mgs, tol_gmres);
    println!("{:?}", x);
    println!("{:?}", a.dot(&x));
    println!("{:?}", res);
}
