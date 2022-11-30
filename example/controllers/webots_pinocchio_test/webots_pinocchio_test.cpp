#include "pinocchio/parsers/urdf.hpp"

#include "pinocchio/algorithm/joint-configuration.hpp"
#include "pinocchio/algorithm/kinematics.hpp"
#include "pinocchio/spatial/explog.hpp"
#include "pinocchio/algorithm/jacobian.hpp"

#include <Eigen/Geometry> 

#include <webots/Motor.hpp>
#include <webots/Robot.hpp>
#include <iostream>
#include <string>
#include <cstdlib>

// model joints
// joint name ::universe 0
// joint name ::shoulder_pan_joint 1
// joint name ::shoulder_lift_joint 2
// joint name ::elbow_joint 3
// joint name ::wrist_1_joint 4
// joint name ::wrist_2_joint 5
// joint name ::wrist_3_joint 6
int main(int argc, char **argv)
{
  using namespace pinocchio;
  // const std::string urdf_filename = std::string("/home/satoshi/darwin_description/darwin.urdf");
  const std::string urdf_filename = std::string("/home/satoshi/repos/pinocchio/models/example-robot-data/robots/ur_description/urdf/ur5_robot.urdf");

  Model model;
  pinocchio::urdf::buildModel(urdf_filename, model);
  std::cout << "model name: " << model.name << std::endl;

  //--------------- webots -----------------------
  constexpr int32_t WEBOTS_TIMESTEP = 8;
  webots::Robot *webots_robot = new webots::Robot();
  std::map<std::string, webots::Motor *> webots_motors;
  for (JointIndex joint_id = 0; joint_id < (JointIndex)model.njoints; ++joint_id)
  {
    std::string joint_name = model.names[joint_id];
    if (joint_name == "universe")
    {
      continue;
    }
    std::cout << "joint name ::" << joint_name << std::endl;
    auto motor_ptr = webots_robot->getMotor(joint_name);
    if (motor_ptr != nullptr)
    {
      webots_motors[joint_name] = motor_ptr;
    }
    else
    {
      std::cerr << "getMotor allocation error !!!" << std::endl;
      std::terminate();
    }
  }
  //--------------- pinocchio construct -----------------------

  // Create data required by the algorithms
  Data data(model);

  // Sample a random configuration
  // Eigen::VectorXd q = randomConfiguration(model);
  Eigen::VectorXd q = pinocchio::neutral(model);
  std::cout << "q: " << q.transpose() << std::endl;

  // Perform the forward kinematics over the kinematic tree
  forwardKinematics(model, data, q);

  // Print out the placement of each joint of the kinematic tree
  for (JointIndex joint_id = 0; joint_id < (JointIndex)model.njoints; ++joint_id)
    std::cout << std::setw(24) << std::left
              << model.names[joint_id] << ": "
              << std::fixed << std::setprecision(2)
              // << data.oMi[joint_id].translation().transpose()
              << data.oMi[joint_id].translation().transpose()
              << std::endl;

  //--------------- pinocchio inverse kinematics -----------------------
  constexpr double eps = 1e-4;
  constexpr int32_t IT_MAX = 1000;
  constexpr double DT = 1e-1;
  constexpr double damp = 1e-6;

  pinocchio::Data::Matrix6x J(6, model.nv);
  J.setZero();
  const int JOINT_ID = 6; // endeffector が連結してるjoint
  double x = 0,y = 0,z = 0;
  double roll = 0,pitch = 0,yaw = 0;
  Eigen::Matrix3d rot ;
  std::cout << " argc is " << argc << std::endl;
  if (argc >= 7){
    x = std::atof(argv[1]);
    y = std::atof(argv[2]);
    z = std::atof(argv[3]);
    roll = std::atof(argv[4]);
    pitch = std::atof(argv[5]);
    yaw = std::atof(argv[6]);
    Eigen::Matrix3d tmp;
    rot = Eigen::AngleAxisd(roll * M_PI, Eigen::Vector3d::UnitX()) 
        * Eigen::AngleAxisd(pitch * M_PI, Eigen::Vector3d::UnitY())
        * Eigen::AngleAxisd(yaw * M_PI, Eigen::Vector3d::UnitZ());
    std::cout << rot << std::endl;
  }
  else if (argc >= 4)
  {
    x = std::atof(argv[1]);
    y = std::atof(argv[2]);
    z = std::atof(argv[3]);
  }
  else
  {
    x = 0.4;
    y = 0.;
    z = 0.1;
    rot = Eigen::Matrix3d::Identity();
  }
  const pinocchio::SE3 oMdes(rot , Eigen::Vector3d(x, y, z));
  Eigen::Matrix<double, 6, 1> err;
  Eigen::VectorXd v(model.nv);
  bool success = false;
  for (int i = 0;; i++)
  {
    pinocchio::forwardKinematics(model, data, q);
    const pinocchio::SE3 dMi = oMdes.actInv(data.oMi[JOINT_ID]);
    err = pinocchio::log6(dMi).toVector();
    if (err.norm() < eps)
    {
      success = true;
      break;
    }
    if (i >= IT_MAX)
    {
      success = false;
      break;
    }
    pinocchio::computeJointJacobian(model, data, q, JOINT_ID, J);
    pinocchio::Data::Matrix6 JJt;
    JJt.noalias() = J * J.transpose();
    JJt.diagonal().array() += damp;
    v.noalias() = -J.transpose() * JJt.ldlt().solve(err);
    q = pinocchio::integrate(model, q, v * DT);
    if (!(i % 10))
      std::cout << i << ": error = " << err.transpose() << std::endl;
  }

  if (success)
  {
    std::cout << "Convergence achieved!" << std::endl;
  }
  else
  {
    std::cout << "\nWarning: the iterative algorithm has not reached convergence to the desired precision" << std::endl;
  }

  std::cout << "\nresult: " << q.transpose() << std::endl;
  std::cout << "\nfinal error: " << err.transpose() << std::endl;
  //--------------------------------------------------------------------

  std::cout << "start steps \n";
  for (int64_t step_cnt = 0; webots_robot->step(WEBOTS_TIMESTEP) != -1; step_cnt++)
  {
    for (JointIndex joint_id = 0; joint_id < (JointIndex)model.njoints; ++joint_id)
    {
      std::string joint_name = model.names[joint_id];
      if (joint_name == "universe")
        continue;
      webots_motors[joint_name]->setPosition(q(joint_id - 1));
    }
  }
}