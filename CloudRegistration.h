#define USE_SVD
#include <vector>
#include <Eigen/Eigen>

using namespace Eigen;
using namespace std;

typedef vector<Vector3d> Cloud;

class CloudRegistration
{
public:

	CloudRegistration();

	bool Registration(
		const Cloud& srcPoints, /*data */ 
		const Cloud& dstPoints, /*model */ 
		Affine3d& trans,/*dst = trans*src */ 
		double& scale);

	double ComputeRMS(const Cloud& srcCloud, const Cloud& dstCloud, const Affine3d& trans_with_scale, vector<double>& residuals);
public:
	vector<double> m_coupleWeights;
	bool m_adjustScale;
	bool m_with_weights;

private:
	bool RegistrationProcedure(
		const Cloud& srcPoints, /*data */ 
		const Cloud& dstPoints, /*model */ 
		Affine3d& trans,/*dst = trans*src */ 
		bool adjustScale, 
		double& scale, 
		bool with_weights, 
		const vector<double>& coupleWeights, 
		double aPrioriScale = 1.0);

};
