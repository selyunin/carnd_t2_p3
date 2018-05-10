/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

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
 *  1. define number of particles
 *  2.
 */

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

	//initialize number of particles
	num_particles = 50;

	default_random_engine gen;
	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);

	for (unsigned idx=0; idx < num_particles; ++idx){
		Particle p;
		p.id = idx;
		p.x = dist_x(gen);
		p.y = dist_y(gen);
		p.theta = dist_theta(gen);
		p.weight = 1.0;

		particles.push_back(p);
	}
	weights.resize(num_particles);
	fill(weights.begin(), weights.end(), 1.0);
	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	default_random_engine gen;

	normal_distribution<double> dist_x(0, std_pos[0]);
	normal_distribution<double> dist_y(0, std_pos[1]);
	normal_distribution<double> dist_theta(0, std_pos[2]);

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
		x_f = x_0 + velocity / yaw_rate * (sin(theta_0 + yaw_rate * delta_t) - sin(theta_0));
		y_f = y_0 + velocity / yaw_rate * (-cos(theta_0 + yaw_rate * delta_t) + cos(theta_0));
		theta_f = theta_0 + yaw_rate * delta_t;
		}

		particles[idx].x = x_f + dist_x(gen);
		particles[idx].y = y_f + dist_y(gen);
		particles[idx].theta = theta_f + dist_theta(gen);
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
	// observations --> reported from the sensor
	// predicted --> associated landmark from the landmark map
	for (unsigned obs_id = 0; obs_id < observations.size(); ++obs_id){
		double running_min = 1e9;
		int running_id = -1;
		double x_obs = observations[obs_id].x;
		double y_obs = observations[obs_id].y;
		for (unsigned pred_id = 0; pred_id< predicted.size(); ++pred_id){
			double x_pred = predicted[pred_id].x;
			double y_pred = predicted[pred_id].y;
			double l2_norm = dist(x_pred, y_pred, x_obs, y_obs);
			if (l2_norm < running_min){
				running_min = l2_norm;
				running_id = predicted[pred_id].id;
			}
		}
		observations[obs_id].id = running_id;
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

	for (unsigned idx=0; idx<num_particles; ++idx){

		double x_p = particles[idx].x;
		double y_p = particles[idx].y;
		double theta = particles[idx].theta;
		//for each particle identify relevant landmarks:
		//i.e. landmarks within the sensor range
		std::vector<LandmarkObs> p_pred;
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
				p_pred.push_back(loc);
			}
		}

		std::vector<LandmarkObs> p_obs;
		for(unsigned obs_idx=0; obs_idx<observations.size();++obs_idx){
			//calculate transformation
			double x_c = observations[obs_idx].x;
			double y_c = observations[obs_idx].y;
			//landmark observations in the map coordinates
			double x_m = x_p + cos(theta) * x_c - sin(theta) * y_c;
			double y_m = y_p + sin(theta) * x_c + cos(theta) * y_c;
			LandmarkObs obs;
			obs.x = x_m;
			obs.y = y_m;
			p_obs.push_back(obs);
		}

		dataAssociation(p_pred, p_obs);

		double p_weight = 1.0;

		for(unsigned idx=0; idx <p_obs.size(); ++idx){
			std::vector<LandmarkObs>::iterator it;
			it = find_if(p_pred.begin(), p_pred.end(), [&] (const LandmarkObs& l) { return l.id == p_obs[idx].id; } );
			double x_landmark = it->x;
			double y_landmark = it->y;

			double x_term = pow(p_obs[idx].x - x_landmark, 2) / (2 * pow(std_landmark[0], 2));
			double y_term = pow(p_obs[idx].y - y_landmark, 2) / (2 * pow(std_landmark[1], 2));

			double w = exp(-(x_term + y_term)) / (2 * M_PI * std_landmark[0] * std_landmark[1]);
		    p_weight *=  w;
		}

		weights[idx] = p_weight;
		particles[idx].weight = p_weight;
	}

}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	  std::default_random_engine generator;

	  std::discrete_distribution<> dist(weights.begin(), weights.end());

	  vector<Particle> new_particles;

	  for(unsigned i=0; i<num_particles; i++){
	    int p_idx = dist(generator);
	    new_particles.push_back(particles[p_idx]);
	  }

	  particles = new_particles;

}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
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
