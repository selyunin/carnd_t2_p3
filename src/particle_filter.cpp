#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;


/*! \brief Initializing particle filter
 *
 *  x, y, theta - vehicle coordinates for initialization
 *  std - standard deviations for x, y, and theta
 *  1. initialize number of particles
 *  2. initialize particles around x,y,theta with a variance std^2
 *  3. initialize all weights to 1.0
 */
void ParticleFilter::init(double x, double y, double theta, double std[]) {
	//1. initialize number of particles
	num_particles = 50;
	//2. initialize particles around x,y,theta with a variance std^2
	//2.1 define normal distributions with std of x, y, and theta
	default_random_engine gen;
	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);
	//2.2 randomly initialize particles around (x,y,theta)
	for (unsigned idx=0; idx < num_particles; ++idx){
		Particle p;
		p.id = idx;
		p.x = dist_x(gen);
		p.y = dist_y(gen);
		p.theta = dist_theta(gen);
		p.weight = 1.0;

		particles.push_back(p);
	}
	//3. initialize all weights to 1.0
	weights.resize(num_particles);
	fill(weights.begin(), weights.end(), 1.0);

	is_initialized = true;
}


/*! \brief Particle filter prediction:
 *         update movement according to the CTRV model
 *
 *  delta_t - time between measurements
 *  std_pos - standard deviations of x,y,theta
 *  velocity - computed velocity
 *  yaw_rate - computed angular velocity
 *
 *  Assuming that the vehicle does not change velocity between measurements
 *  (CTRV assumption) we update positions of the particles w.r.t. the motion model.
 *  We compute the final points, then we sample from Gaussians with the corresponding std:
 *  1. initialize Gaussian distributions for x, y, theta with std_x, std_y, std_theta
 *  2. compute particle motion  with CTRV assumptions
 *  3. update particle positions by drawing sample from the Gaussian around the motion
 */
void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// 1. initialize Gaussian distributions for x, y, theta with std_x, std_y, std_theta
	default_random_engine gen;

	normal_distribution<double> dist_x(0, std_pos[0]);
	normal_distribution<double> dist_y(0, std_pos[1]);
	normal_distribution<double> dist_theta(0, std_pos[2]);
	// 2. compute particle motion  with CTRV assumptions
	for (unsigned idx = 0; idx < num_particles; ++idx){
		double x_0 = particles[idx].x;
		double y_0 = particles[idx].y;
		double theta_0 = particles[idx].theta;
		double x_f, y_f, theta_f;
		if (fabs(yaw_rate) < 1e-4){
			//assuming moving straight
			x_f = x_0 + velocity * cos(theta_0) * delta_t;
			y_f = y_0 + velocity * sin(theta_0) * delta_t;
			theta_f = theta_0;
		} else {
		// CTRV update
		x_f = x_0 + velocity / yaw_rate * (sin(theta_0 + yaw_rate * delta_t) - sin(theta_0));
		y_f = y_0 + velocity / yaw_rate * (-cos(theta_0 + yaw_rate * delta_t) + cos(theta_0));
		theta_f = theta_0 + yaw_rate * delta_t;
		}
		// 3. update particle positions by drawing sample from the Gaussian around the motion
		particles[idx].x = x_f + dist_x(gen);
		particles[idx].y = y_f + dist_y(gen);
		particles[idx].theta = theta_f + dist_theta(gen);
	}
}


/*! \brief dataAssociation:
 *         identify closest landmark to the measurement
 *
 *  map_landmark_coordinates - coordinates of landmarks from the map
 *  observations - observations from sensor (in map coordinates)
 *
 *  Given landmark coordinates from the map and the actual sensor data,
 *  identify the closest landmark to the measurements.
 *  Both in map landmarks and observations are in the map coordinates
 *  1. for each observation, iterate over all map landmarks and find the minimum distance
 *  2. the closest landmark id is assigned to the observation
 */
void ParticleFilter::dataAssociation(const std::vector<LandmarkObs>& map_landmark_coordinates, std::vector<LandmarkObs>& observations) {
	for (unsigned obs_id = 0; obs_id < observations.size(); ++obs_id){
		double running_min = 1e9;
		int running_id = -1;
		double x_obs = observations[obs_id].x;
		double y_obs = observations[obs_id].y;
		// 1. for each observation, iterate over all map landmarks and find the minimum distance
		for (unsigned pred_id = 0; pred_id< map_landmark_coordinates.size(); ++pred_id){
			double x_pred = map_landmark_coordinates[pred_id].x;
			double y_pred = map_landmark_coordinates[pred_id].y;
			double l2_norm = dist(x_pred, y_pred, x_obs, y_obs);
			if (l2_norm < running_min){
				running_min = l2_norm;
				running_id = map_landmark_coordinates[pred_id].id;
			}
		}
		// 2. the closest landmark id is assigned to the observation
		observations[obs_id].id = running_id;
	}
}


/*! \brief updateWeights:
 *         update particle weight depending how plausible its position is
 *
 *  sensor_range - max distance the sensor sees the landmarks
 *  std_landmark - std components of the MGD
 *  observations - vector of sensor observations
 *  map_landmarks - map with the landmarks
 *
 *  1. given the position of the particle, and the sensor range,
 *     find landmarks which particle could possibly sense
 *  2. homogeneous transformation of observations from
 *     local vehicle coordinates to the map coordinates
 *  3. associate landmarks within the particle sensor range (found in 1) with
 *     the sensor measurements
 *  4. for each observation, calculate MGD(x,y) points of associated sensor measurements
 *     and the actual landmark coordinates.
 *     The updated weight for each particle is a product of probabilities for all observations
 */
void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {

	for (unsigned idx=0; idx<num_particles; ++idx){
		double x_p = particles[idx].x;
		double y_p = particles[idx].y;
		double theta = particles[idx].theta;


		//  1. given the position of the particle (x_p, y_p), and the sensor range,
		//     find landmarks which the particle could possibly sense
		std::vector<LandmarkObs> landmarks_within_range;
		for (unsigned map_id=0; map_id<map_landmarks.landmark_list.size(); ++map_id){
			//landmark coordinates
			double x_l = map_landmarks.landmark_list[map_id].x_f;
			double y_l = map_landmarks.landmark_list[map_id].y_f;
			double distance_to_landmark = dist(x_p, y_p, x_l, y_l);
			if (distance_to_landmark <= sensor_range){
				//
				LandmarkObs loc;
				loc.id = map_landmarks.landmark_list[map_id].id_i;
				loc.x = map_landmarks.landmark_list[map_id].x_f;
				loc.y = map_landmarks.landmark_list[map_id].y_f;
				landmarks_within_range.push_back(loc);
			}
		}
		//  2. homogeneous transformation of observations from
		//     local vehicle coordinates to the map coordinates
		std::vector<LandmarkObs> observations_map_coords;
		for(unsigned obs_idx=0; obs_idx<observations.size();++obs_idx){
			double x_c = observations[obs_idx].x;
			double y_c = observations[obs_idx].y;
			//landmark observations in the map coordinates
			double x_m = x_p + cos(theta) * x_c - sin(theta) * y_c;
			double y_m = y_p + sin(theta) * x_c + cos(theta) * y_c;
			LandmarkObs obs;
			obs.x = x_m;
			obs.y = y_m;
			observations_map_coords.push_back(obs);
		}
		//  3. associate landmarks within the particle sensor range (found in 1.) with
		//     the sensor measurements
		dataAssociation(landmarks_within_range, observations_map_coords);

		//  4. for each observation, calculate MGD(x,y) points of associated sensor measurements
		//     and the actual landmark coordinates.
		double p_weight = 1.0;
		for(unsigned idx=0; idx <observations_map_coords.size(); ++idx){
			// find landmark coordinates which was associated with this measurement
			std::vector<LandmarkObs>::iterator it;
			it = find_if(landmarks_within_range.begin(), landmarks_within_range.end(),
					     [&] (const LandmarkObs& l) { return l.id == observations_map_coords[idx].id; } );
			double x_landmark = it->x;
			double y_landmark = it->y;
			double x_obs = observations_map_coords[idx].x;
			double y_obs = observations_map_coords[idx].y;
			// use MGD to find the probability of the association
			double p_mgd = 1.0 / (2.0 * M_PI * std_landmark[0] * std_landmark[1]) *
					   exp(-(pow(x_obs - x_landmark, 2) / (2 * pow(std_landmark[0], 2)) +
					 	     pow(y_obs - y_landmark, 2) / (2 * pow(std_landmark[1], 2))));
			// The updated weight for each particle is a product of probabilities for all observations
			p_weight *=  p_mgd;
		}
		weights[idx] = p_weight;
		particles[idx].weight = p_weight;
	}

}

/*! \brief resample:
 *         resample particles with replacement with probability proportional to their weight.
 *
 *	 std::discrete_distribution distribution produces integer values s.t.
 *	 each possible value has a probability of being produced is proportional to the weight
 *	 This is exactly what we need!
 *	 In general, the weights do not need to be normalized to [0,1].
 */
void ParticleFilter::resample() {

	  default_random_engine generator;

	  discrete_distribution<unsigned> dist(weights.begin(), weights.end());

	  vector<Particle> new_particles;

	  for(unsigned i=0; i<num_particles; i++){
	    unsigned p_idx = dist(generator);
	    new_particles.push_back(particles[p_idx]);
	  }

	  particles = new_particles;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    // particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations = associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;

    return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
