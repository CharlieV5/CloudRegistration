#include "CloudRegistration.h"
#include <vector>
#include <Eigen/Eigen>

using namespace Eigen;
using namespace std;


void GetBoundingBox(const Cloud& cloud, Vector3d& bbMin, Vector3d& bbMax)
{
	if (cloud.size() == 0)
		return;
	
	double dMax = numeric_limits<double>::max();
	double dMin = -dMax;

	bbMin = Vector3d(dMax, dMax, dMax);
	bbMax = Vector3d(dMin, dMin, dMin);
	for (auto& p : cloud)
	{
		if (p.x() > bbMax.x())
		{
			bbMax.x() = p.x();
		}
		if (p.y() > bbMax.y())
		{
			bbMax.y() = p.y();
		}
		if (p.z() > bbMax.z())
		{
			bbMax.z() = p.z();
		}

		if (p.x() < bbMin.x())
		{
			bbMin.x() = p.x();
		}
		if (p.y() < bbMin.y())
		{
			bbMin.y() = p.y();
		}
		if (p.z() < bbMin.z())
		{
			bbMin.z() = p.z();
		}
	}
}

bool ComputeGravityCenter(const Cloud& cloud, Vector3d& center)
{
	unsigned count = cloud.size();
	if (count == 0)
		return false;

	center.setZero();

	for (const auto& p:cloud)
	{
		center += p;
	}

	center /= count;

	return true;
}

bool ComputeWeightedGravityCenter(const Cloud& cloud, const vector<double>& weights, Vector3d& center)
{
	unsigned count = cloud.size();
	if (count == 0 || weights.size() != count)
		return false;

	center.setZero();

	double wSum = 0;
	for (unsigned i = 0; i < count; ++i)
	{
		const Vector3d& p = cloud[i];
		double w = std::fabs(weights[i]);
		center += p * w;
		wSum += w;
	}

	if (wSum != 0)
		center /= wSum;

	return true;
}

Matrix3d ComputeCrossCovarianceMatrix(const Cloud& P, const Cloud& Q, const Vector3d& Gp, const Vector3d& Gq)
{
	assert(Q.size() == P.size());

	//shortcuts to output matrix lines
	Matrix3d covMat;
	//sums
	unsigned count = P.size();
	for (unsigned i = 0; i < count; i++)
	{
		Vector3d Pt = P[i] - Gp;        // 3¡Á1
		Vector3d Qt = Q[i] - Gq;        // 3¡Á1
		auto Qt_T = Qt.transpose();     // 1¡Á3

		covMat += Pt * Qt_T;

		//covMat(0, 0) += Pt.x() * Qt.x();
		//covMat(0, 1) += Pt.x() * Qt.y();
		//covMat(0, 2) += Pt.x() * Qt.z();
		//covMat(1, 0) += Pt.y() * Qt.x();
		//covMat(1, 1) += Pt.y() * Qt.y();
		//covMat(1, 2) += Pt.y() * Qt.z();
		//covMat(2, 0) += Pt.z() * Qt.x();
		//covMat(2, 1) += Pt.z() * Qt.y();
		//covMat(2, 2) += Pt.z() * Qt.z();
	}

	covMat = (1.0 / count)*covMat;

	return covMat;
}

Matrix3d ComputeWeightedCrossCovarianceMatrix(
	const Cloud& P, //data
	const Cloud& Q, //model
	const Vector3d& Gp,
	const Vector3d& Gq,
	const vector<double>& coupleWeights)
{
	assert(Q.size() == P.size());
	assert(coupleWeights.size() == P.size());

	//shortcuts to output matrix lines
	Matrix3d covMat = Matrix3d::Zero();

	//sums
	unsigned count = P.size();
	double wSum = 0.0; //we will normalize by the sum
	for (unsigned i = 0; i < count; i++)
	{
		const Vector3d& Pt = P[i];     // 3¡Á1
		const Vector3d& Qt = Q[i];     // 3¡Á1
		auto Qt_T = Qt.transpose();    // 1¡Á3
		//Weighting scheme for cross-covariance is inspired from
		//https://en.wikipedia.org/wiki/Weighted_arithmetic_mean#Weighted_sample_covariance
		double wi = fabs(coupleWeights[i]);

		//DGM: we virtually make the P (data) point nearer if it has a lower weight
		Vector3d Ptw = wi*Pt;
		wSum += wi;

		covMat += Ptw * Qt_T;

		////1st row
		//covMat(0, 0) += Ptw.x() * Qt.x();
		//covMat(0, 1) += Ptw.x() * Qt.y();
		//covMat(0, 2) += Ptw.x() * Qt.z();
		////2nd row	  		 
		//covMat(1, 0) += Ptw.y() * Qt.x();
		//covMat(1, 1) += Ptw.y() * Qt.y();
		//covMat(1, 2) += Ptw.y() * Qt.z();
		////3rd row	  		 
		//covMat(2, 0) += Ptw.z() * Qt.x();
		//covMat(2, 1) += Ptw.z() * Qt.y();
		//covMat(2, 2) += Ptw.z() * Qt.z();
	}

	if (wSum != 0.0)
	{
		covMat = (1.0 / wSum)*covMat;
	}

	auto Gq_T = Gq.transpose();
	covMat -= Gp * Gq_T;

	//remove the centers of gravity
	//covMat(0, 0) -= Gp.x() * Gq.x();
	//covMat(0, 1) -= Gp.x() * Gq.y();
	//covMat(0, 2) -= Gp.x() * Gq.z();
	////2nd row			 
	//covMat(1, 0) -= Gp.y() * Gq.x();
	//covMat(1, 1) -= Gp.y() * Gq.y();
	//covMat(1, 2) -= Gp.y() * Gq.z();
	////3rd row			 
	//covMat(2, 0) -= Gp.z() * Gq.x();
	//covMat(2, 1) -= Gp.z() * Gq.y();
	//covMat(2, 2) -= Gp.z() * Gq.z();

	return covMat;
}


bool LessThanEpsilon(double a)
{
	return a < numeric_limits<double>::epsilon();
}

CloudRegistration::CloudRegistration()
{
	m_coupleWeights.clear();
	m_adjustScale = false;
	m_with_weights = false;
}

bool CloudRegistration::RegistrationProcedure(const Cloud& srcPoints, //data
	const Cloud& dstPoints, //model
	Affine3d& trans,//dst = trans*src
	bool adjustScale/*=false*/,
	double& scale,
	bool with_weights/*=false*/,
	const vector<double>& coupleWeights,
	double aPrioriScale/*=1.0f*/)
{

	if (srcPoints.size() != dstPoints.size() || srcPoints.size() < 3)
		return false;

	//centers of mass
	Vector3d Gp, Gx;
	with_weights ? ComputeWeightedGravityCenter(srcPoints, coupleWeights, Gp) : ComputeGravityCenter(srcPoints, Gp);
	with_weights ? ComputeWeightedGravityCenter(dstPoints, coupleWeights, Gx) : ComputeGravityCenter(dstPoints, Gx);

	//specific case: 3 points only
	//See section 5.A in Horn's paper
	if (srcPoints.size() == 3)
	{
		//compute the first set normal
		const Vector3d& Ap = srcPoints[0];
		const Vector3d& Bp = srcPoints[1];
		const Vector3d& Cp = srcPoints[2];
		Vector3d Np(0, 0, 1);
		{
			Np = (Bp - Ap).cross(Cp - Ap);
			double norm = Np.norm();
			if (LessThanEpsilon(norm))
			{
				return false;
			}
			Np /= norm;
		}
		//compute the second set normal
		const Vector3d& Ax = dstPoints[0];
		const Vector3d& Bx = dstPoints[1];
		const Vector3d& Cx = dstPoints[2];
		Vector3d Nx(0, 0, 1);
		{
			Nx = (Bx - Ax).cross(Cx - Ax);
			double norm = Nx.norm();
			if (LessThanEpsilon(norm))
			{
				return false;
			}
			Nx /= norm;
		}
		//now the rotation is simply the rotation from Nx to Np, centered on Gx
		Vector3d a = Np.cross(Nx);
		if (LessThanEpsilon(a.norm()))
		{
			trans.setIdentity();
			if (Np.dot(Nx) < 0)
			{
				trans.scale(-1);
			}
		}
		else
		{
			double cos_t = Np.dot(Nx);
			assert(cos_t > -1.0 && cos_t < 1.0); //see above
			double s = sqrt((1 + cos_t) * 2);
			double q[4] = { s / 2, a.x() / s, a.y() / s, a.z() / s }; //don't forget to normalize the quaternion
			double qnorm = q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3];
			assert(!LessThanEpsilon(qnorm));
			qnorm = sqrt(qnorm);
			q[0] /= qnorm;// w
			q[1] /= qnorm;// x
			q[2] /= qnorm;// y
			q[3] /= qnorm;// z

			trans.rotate(Quaterniond(q[0], q[1], q[2], q[3]));
		}

		if (adjustScale)
		{
			double sumNormP = (Bp - Ap).norm() + (Cp - Bp).norm() + (Ap - Cp).norm();
			sumNormP *= aPrioriScale;
			if (sumNormP < numeric_limits<double>::epsilon())
			{
				return false;
			}
			double sumNormX = (Bx - Ax).norm() + (Cx - Bx).norm() + (Ax - Cx).norm();
			scale = (sumNormX / sumNormP); //sumNormX / (sumNormP * Sa) in fact
		}

		//we deduce the first translation
		trans.translation() = Gx - (trans.rotation() * Gp) * (aPrioriScale * scale);
		//#26 in besl paper, modified with the scale as in jschmidt

		//we need to find the rotation in the (X) plane now
		{
			auto App = trans*Ap;
			auto Bpp = trans*Bp;
			auto Cpp = trans*Cp;

			double C = 0;
			double S = 0;
			Vector3d Ssum(0, 0, 0);
			Vector3d rx;
			Vector3d rp;

			rx = Ax - Gx;
			rp = App - Gx;
			C = rx.dot(rp);
			Ssum = rx.cross(rp);

			rx = Bx - Gx;
			rp = Bpp - Gx;
			C += rx.dot(rp);
			Ssum += rx.cross(rp);

			rx = Cx - Gx;
			rp = Cpp - Gx;
			C += rx.dot(rp);
			Ssum += rx.cross(rp);

			S = Ssum.dot(Nx);
			double Q = sqrt(S * S + C * C);
			if (LessThanEpsilon(Q))
			{
				return false;
			}

			double sin_t = (S / Q);
			double cos_t = (C / Q);
			double inv_cos_t = 1 - cos_t;

			double l1 = Nx.x();
			double l2 = Nx.y();
			double l3 = Nx.z();

			double l1_inv_cos_t = l1 * inv_cos_t;
			double l3_inv_cos_t = l3 * inv_cos_t;

			Matrix3d R;
			//1st column
			R(0, 0) = cos_t + l1 * l1_inv_cos_t;
			R(1, 0) = l2 * l1_inv_cos_t + l3 * sin_t;
			R(2, 0) = l3 * l1_inv_cos_t - l2 * sin_t;

			//2nd column
			R(0, 1) = l2 * l1_inv_cos_t - l3 * sin_t;
			R(1, 1) = cos_t + l2 * l2 * inv_cos_t;
			R(2, 1) = l2 * l3_inv_cos_t + l1 * sin_t;

			//3rd column
			R(0, 2) = l3 * l1_inv_cos_t + l2 * sin_t;
			R(1, 2) = l2 * l3_inv_cos_t - l1 * sin_t;
			R(2, 2) = cos_t + l3 * l3_inv_cos_t;

			trans.linear() = R * trans.linear();
			trans.translation() = Gx - (trans.linear() * Gp) * (aPrioriScale * scale); //update T as well
		}
	}
	else
	{
		Vector3d bbMin;
		Vector3d bbMax;
		//refPoints->getBoundingBox(bbMin, bbMax);
		GetBoundingBox(dstPoints, bbMin, bbMax);

		//if the data cloud is equivalent to a single point (for instance
		//it's the case when the two clouds are very far away from
		//each other in the ICP process) we try to get the two clouds closer
		Vector3d diag = bbMax - bbMin;
		if (LessThanEpsilon(fabs(diag.x()) + fabs(diag.y()) + fabs(diag.z())))
		{
			trans.translation() = (Gx - Gp * aPrioriScale);
			return true;
		}

		//Cross covariance matrix, eq #24 in Besl92 (but with weights, if any)
		Matrix3d Sigma_px = (with_weights ? 
			ComputeWeightedCrossCovarianceMatrix(srcPoints, dstPoints, Gp, Gx, coupleWeights)
			: ComputeCrossCovarianceMatrix(srcPoints, dstPoints, Gp, Gx));

		JacobiSVD<MatrixXd> svd(Sigma_px, ComputeFullU | ComputeFullV);
		MatrixXd U = svd.matrixU();
		MatrixXd V = svd.matrixV();

		MatrixXd UT = U.transpose();
		trans.linear() = V * UT;

		if (adjustScale)
		{
			//two accumulators
			double acc_num = 0.0;
			double acc_denom = 0.0;

			//now deduce the scale (refer to "Point Set Registration with Integrated Scale Estimation", Zinsser et. al, PRIP 2005)
			unsigned count = dstPoints.size();
			assert(srcPoints.size() == count);
			for (unsigned i = 0; i < count; ++i)
			{
				//'a' refers to the data 'A' (moving) = P
				//'b' refers to the model 'B' (not moving) = X
				Vector3d a_tilde = trans.linear() * (srcPoints[i] - Gp);	// a_tilde_i = R * (a_i - a_mean)
				Vector3d b_tilde = dstPoints[i] - Gx;			// b_tilde_j =     (b_j - b_mean)

				acc_num += b_tilde.dot(a_tilde);
				acc_denom += a_tilde.dot(a_tilde);
			}

			//DGM: acc_2 can't be 0 because we already have checked that the bbox is not a single point!
			assert(acc_denom > 0.0);
			scale = fabs(acc_num / acc_denom);
		}

		//and we deduce the translation
		trans.translation() = Gx - (trans.linear() * Gp) * (aPrioriScale * scale); //#26 in besl paper, modified with the scale as in jschmidt
	}

	return true;
}

bool CloudRegistration::Registration(const Cloud& srcPoints, /*data */ const Cloud& dstPoints, /*model */ Affine3d& trans,/*dst = trans*src */ double& scale)
{
	return RegistrationProcedure(
		srcPoints, //data
		dstPoints, //model
		trans,//dst = trans*src
		m_adjustScale,
		scale,
		m_with_weights,
		m_coupleWeights
		);
}


double CloudRegistration::ComputeRMS(
	const Cloud& srcCloud,
	const Cloud& dstCloud,
	const Affine3d& trans_with_scale,
	vector<double>& residuals)
{
	if (dstCloud.size() != srcCloud.size() || dstCloud.size() < 3)
		return -1;

	double rms = 0.0;

	unsigned count = dstCloud.size();
	residuals.resize(count);
	for (unsigned i = 0; i < count; i++)
	{
		const Vector3d& Ri = dstCloud[i];
		const Vector3d& Li = srcCloud[i];
		Vector3d Lit = trans_with_scale *Li;

		residuals[i] = (Ri - Lit).squaredNorm();

		rms += residuals[i];
	}

	return sqrt(rms / count);
}