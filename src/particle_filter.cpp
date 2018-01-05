/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <limits>
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

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  num_particles = 500;

  particles = vector<Particle>(num_particles);
  weights = vector<double>(num_particles);

  default_random_engine gen;
  normal_distribution<double> dist_x(x,std[0]);
  normal_distribution<double> dist_y(y,std[1]);
  normal_distribution<double> dist_theta(theta,std[2]);

  double w = 1.0 / num_particles;

  for (int i = 0; i < num_particles; i++){
    particles.at(i).id = i;
    particles.at(i).x = dist_x(gen);
    particles.at(i).y = dist_y(gen);
    particles.at(i).theta = dist_theta(gen);
    particles.at(i).weight = w;
    weights.at(i) = w;
  }
  
  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
  default_random_engine gen;
  normal_distribution<double> dist_x(0.0,std_pos[0]);
  normal_distribution<double> dist_y(0.0,std_pos[1]); 
  normal_distribution<double> dist_theta(0.0,std_pos[2]);
 
  for (int i =0; i < num_particles; i++){
    double theta = particles.at(i).theta;
    if(fabs(yaw_rate) > 0.001){
      particles.at(i).x += velocity * ( sin(theta + yaw_rate*delta_t) - sin(theta) ) / yaw_rate;
      particles.at(i).y += velocity * ( -cos(theta + yaw_rate*delta_t) +cos(theta) ) / yaw_rate;
      particles.at(i).theta += yaw_rate * delta_t;
    } else {
      particles.at(i).x += velocity * cos(theta) * delta_t;
      particles.at(i).y += velocity * sin(theta) * delta_t;
    }

    particles.at(i).x += dist_x(gen);
    particles.at(i).y += dist_y(gen);
    particles.at(i).theta += dist_theta(gen);
  }
}

std::vector<int> ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs> observations) {
  std::vector<int> association;
  for(int i = 0; i < observations.size(); i++){
    double min_dis = numeric_limits<double>::max();
    int asso = -1;
    for(int j = 0; j < predicted.size(); j++){
      double distance = dist(observations.at(i).x,observations.at(i).y,predicted.at(j).x,predicted.at(j).y);
      if (distance < min_dis) {
	min_dis = distance;
        asso = j;
      }
    }
    association.push_back(asso);
  }
  return association;
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
  for (int i = 0; i < num_particles; i++){
    int num_obs = observations.size();

    std::vector<LandmarkObs> map_obs = vector<LandmarkObs>(observations.size());
    for (int j = 0; j < num_obs; j++){
      map_obs.at(j).id = observations.at(j).id;
      double cos_th = cos(particles.at(i).theta);
      double sin_th = sin(particles.at(i).theta);
      map_obs.at(j).x = particles.at(i).x + observations.at(j).x * cos_th - observations.at(j).y * sin_th;
      map_obs.at(j).y = particles.at(i).y + observations.at(j).x * sin_th + observations.at(j).y * cos_th;
    }

    std::vector<LandmarkObs> pred;
    for (int j = 0; j < map_landmarks.landmark_list.size(); j++){
      auto mark = map_landmarks.landmark_list.at(j);
      double distance = dist(particles.at(i).x,particles.at(i).y,mark.x_f,mark.y_f);
      if (distance < sensor_range) {
        LandmarkObs pred_mark;
	    pred_mark.id = mark.id_i;
	    pred_mark.x = mark.x_f;
	    pred_mark.y = mark.y_f;
        pred.push_back(pred_mark);
	  }
    }
    
    std::vector<int> association = dataAssociation(pred,map_obs);

    double probability = 1.0;
    for (int j = 0; j < num_obs; j++){
      double dx = map_obs.at(j).x - pred.at(association.at(j)).x;
      double dy = map_obs.at(j).y - pred.at(association.at(j)).y;
      double exponential = - dx*dx / (2*std_landmark[0]*std_landmark[0]) - dy*dy / (2*std_landmark[1]*std_landmark[1]);
      probability *= exp(exponential) / (2 * M_PI * std_landmark[0] * std_landmark[1]);
    }
    particles.at(i).weight = probability;
    weights.at(i) = probability;
  }
}

void ParticleFilter::resample() {
  default_random_engine gen;
  discrete_distribution<int> dist(weights.begin(),weights.end());
  std::vector<Particle> new_particles = vector<Particle>(num_particles);
  for(int i = 0; i < num_particles; i++) new_particles.at(i) = particles.at(dist(gen));
  particles = new_particles;
}

void ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations,
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
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
