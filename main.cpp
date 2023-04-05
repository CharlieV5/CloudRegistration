#include <iostream>
#include "CloudRegistration.h"

int main()
{
	Cloud dstPoints = {
		Vector3d(13.8200, -2.9600, 3.7707),
		Vector3d(11.4067, -6.7387, 3.3698),
		Vector3d(17.0328, -5.2574, 4.9653),
		Vector3d(14.6674, -8.2601, 5.1288)};

	Cloud srcPoints = {
		Vector3d(22.44, 0.28,3.74),
		Vector3d(20.78,-3.49,1.94),
		Vector3d(24.29,-2.02,6.62),
		Vector3d(22.32,-5.027,5.3) };

	Affine3d trans;
	double scale = 1.0;
	CloudRegistration reg;

	reg.m_adjustScale = false;

	reg.Registration(srcPoints, dstPoints, trans, scale);

	trans.scale(scale);

	vector<double> residuals;
	double rms = reg.ComputeRMS(srcPoints, dstPoints, trans, residuals);

	cout << "transform matrix with scale\n" << trans.matrix() << endl << "scale " << scale << endl << "rms " << rms << endl;
	system("pause");
}
