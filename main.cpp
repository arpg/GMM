#include <fstream>

#include "GMM.h"
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Geometry>
#include <unsupported/Eigen/MatrixFunctions>

static Eigen::Matrix3d getSkewSymm(Eigen::Vector3d& t)
{
	Eigen::Matrix3d t_hat;
	t_hat << 0, -t(2), t(1),
		t(2), 0, -t(0),
		-t(1), t(0), 0;
	return t_hat;
}

int main(){
	std::string type_covariance = "full"; // <-> "diagonal"

	int dimension_data = 7;
	int number_data = 10;
	int number_iterations = 50;

	int number_gaussian_components = 2;

	double **data = new double*[number_data];

	std::ofstream file;

	Gaussian_Mixture_Model GMM = Gaussian_Mixture_Model(type_covariance, dimension_data, number_gaussian_components);

	for (int i = 0; i < number_data; i++){
		double position[] = { 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0 };
		position[3] *= pow(-1,i%2);

		Eigen::Quaterniond q(position[3],position[4],position[5],position[6]);
		Eigen::Matrix3d R = q.toRotationMatrix();
		// fprintf(stdout, "R %f %f %f\n", R(0), R(1), R(2));
		// fprintf(stdout, "  %f %f %f\n", R(3), R(4), R(5));
		// fprintf(stdout, "  %f %f %f\n", R(6), R(7), R(8));

		Eigen::Vector3d noise(0.1 * rand() / RAND_MAX - 0.05, 0.1 * rand() / RAND_MAX - 0.05, 0.1 * rand() / RAND_MAX - 0.05);
		// fprintf(stdout, "noise %f %f %f\n", noise[0], noise[1], noise[2]);
		Eigen::Matrix3d noise_mat = getSkewSymm(noise);
		// fprintf(stdout, "noise_mat %f %f %f\n", noise_mat(0), noise_mat(1), noise_mat(2));
		// fprintf(stdout, "          %f %f %f\n", noise_mat(3), noise_mat(4), noise_mat(5));
		// fprintf(stdout, "          %f %f %f\n", noise_mat(6), noise_mat(7), noise_mat(8));
		Eigen::Matrix3d noisy_mat_exp = noise_mat.exp();
		Eigen::Matrix3d noisy_R = R*noisy_mat_exp;
		// fprintf(stdout, "noisy_R %f %f %f\n", noisy_R(0), noisy_R(1), noisy_R(2));
		// fprintf(stdout, "        %f %f %f\n", noisy_R(3), noisy_R(4), noisy_R(5));
		// fprintf(stdout, "        %f %f %f\n", noisy_R(6), noisy_R(7), noisy_R(8));
		Eigen::Quaterniond noisy_q(noisy_R);
		// noisy_q.normalize();

		data[i] = new double[dimension_data];
		data[i][0] = 0.25 * rand() / RAND_MAX - 0.125 + position[0];
		data[i][1] = 0.25 * rand() / RAND_MAX - 0.125 + position[1];
		data[i][2] = 0.25 * rand() / RAND_MAX - 0.125 + position[2];
		data[i][3] = noisy_q.w();
		data[i][4] = noisy_q.x();
		data[i][5] = noisy_q.y();
		data[i][6] = noisy_q.z();

		fprintf(stdout, "loaded %f %f %f %f %f %f %f\n", data[i][0], data[i][1], data[i][2], data[i][3], data[i][4], data[i][5], data[i][6]);

		// double nrm = sqrt(pow(data[i][3],2)+pow(data[i][4],2)+pow(data[i][5],2)+pow(data[i][6],2));
		// data[i][3] /= nrm;
		// data[i][4] /= nrm;
		// data[i][5] /= nrm;
		// data[i][6] /= nrm;
	}

	printf("step	log_likelihood\n");
	for (int i = 0; i < number_iterations; i++){
		double log_likelihood;

		if (i == 0) GMM.Initialize(number_data, data);

		log_likelihood = GMM.Expectaion_Maximization(number_data, data);
		if ((i + 1) % 10 == 0) printf("%d	%lf\n", i + 1, log_likelihood);
	}

	printf("\nmean\n");
	for (int i = 0; i < number_gaussian_components; i++){
		for (int j = 0; j < dimension_data; j++){
			printf("%lf ", GMM.mean[i][j]);
		}
		printf("\n");
	}

	file.open("result.txt");

	for (int j = 0; j < number_gaussian_components; j++){
		for (int i = 0; i < number_data; i++){
			if (GMM.Classify(data[i]) == j){
				file << GMM.Classify(data[i]) << " " << data[i][0] << " " << data[i][1] << endl;
			}
		}
	}
	file.close();

	for (int i = 0; i < number_data; i++){
		delete[] data[i];
	}
	delete[] data;

	return 0;
}
