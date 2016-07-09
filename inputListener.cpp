/**
 * @description
 * Part of a rework of the CoPilot Module.
 * Sets WC velocity from chair joystick, controller joystick, and button inputs from a controller
 *
 * @author Chris Laporte and changed by Lili @July 2015
 * @date   June 2015
 */

#include "JoyControl/Util.h"
#include <JoyControl/joystick.h>
#include <ros/ros.h>
#include <algorithm>
#include <std_msgs/Bool.h>
#include <std_msgs/Empty.h>
#include <std_msgs/Int32.h>
#include <sensor_msgs/Joy.h>
#include <string>

#include <string>
#include <fstream>
#include <sstream>
#include <ostream>


#include <stdio.h>
//#include "SCRF_util.hpp"
#include <iomanip>



#include <stdio.h>
#include <vector>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <iostream>




class readData{
public:
    readData(std::string filename);
    std::vector<std::vector<double> > allDataPointsVec;
    std::vector<double> time;
    std::vector<double> lr_joy;
    std::vector<double> fb_joy;
    std::vector<double> linear_vel;
    std::vector<double> angular_vel;
};

readData::readData(std::string filename)
{
    std::ifstream fin(filename.c_str(), std::ios::in);
    if(!fin.is_open())
    {
        std::cout<<"cannot open file"<<std::endl;
    }

    std::istringstream istr;

    double oneDimension;
    std::vector<double> dataPointVec;
    std::string str;

     while(getline(fin,str))
     {
        istr.str(str);
        while(istr>>oneDimension)
        {
            //cout<<oneDimension<<endl;
            dataPointVec.push_back(oneDimension);
        }
        allDataPointsVec.push_back(dataPointVec);
        dataPointVec.clear();
        istr.clear();
        str.clear();
     }
     fin.close();

    int numOfDimensions=allDataPointsVec[0].size();
    int numOfElements=allDataPointsVec.size();

    for(int i=0; i<numOfElements; i++)
    {
        //cout<<" number of elements is "<<numOfElements<<endl;

      //std::cout<<"The joystick values "<<i<<allDataPointsVec[i][0]<<std::endl;
            //lr_joy.push_back(allDataPointsVec[i][0]);
            //fb_joy.push_back(allDataPointsVec[i][1]);
        time.push_back(allDataPointsVec[i][0]);
        fb_joy.push_back(allDataPointsVec[i][2]);
        lr_joy.push_back(allDataPointsVec[i][1]);
        linear_vel.push_back(allDataPointsVec[i][3]);
        angular_vel.push_back(allDataPointsVec[i][4]);


    }

}


// WheelChair state variables
static bool directional_control = false;
static int top_speed = 4;
static bool can_drive = true;
static bool controller_override = false;
static bool pod_racing_mode = false;

static const double speed_limits[5] = {0.0, 0.25, 0.5, 0.75, 1};

// Variables to keep track of joystick states
static PolarCoordinates ps3joystick;
static PolarCoordinates wcjoystick;
static double joystickLin;
static double controllerLin;

// Messages
static JoyControl::joystick joystickmsg;

// Publishers
boost::shared_ptr<ros::Publisher> joy_output_pub;
boost::shared_ptr<ros::Publisher> limit_pub;
boost::shared_ptr<ros::Publisher> dirCtrl_pub;
boost::shared_ptr<ros::Publisher> eStop_pub;


//**********************************************************
//               MESSAGE PUBLISHERS
//**********************************************************

void publishJoyOutputMessages(JoyControl::joystick joystickmsg) {
	ros::Time now = ros::Time::now();
	joystickmsg.t = now;
	joy_output_pub->publish(joystickmsg);
}

void publishJoyOutputMessages() {
	ros::Time now = ros::Time::now();
	joystickmsg.t = now;
	joy_output_pub->publish(joystickmsg);
}

void publishLimitMessage(double limit) {
	std_msgs::Int32 msg;
	msg.data = limit * 100;
	limit_pub->publish(msg);
}

void publishDirectionalControlStatusMessage(bool isDirectional) {
	std_msgs::Bool msg;
	msg.data = isDirectional;
	dirCtrl_pub->publish(msg);
}

void publishEStopMessage(bool canDrive) {
	std_msgs::Bool msg;
	msg.data = !can_drive;
	eStop_pub->publish(msg);
}

//***************************************************
//
//       FUNTIONS FOR POD RACING MODE (WC USER MUST HAVE BOTH JOYSTICK AND CONTROLLER)
//
//       joy should be in left hand, controller in right hand
//
//***************************************************

void setPodRacingSpeed() {
	double linear;
	double angular;
	angular = (controllerLin - joystickLin)*2;
	if(controllerLin > 0 && joystickLin > 0) {
		linear = std::min(controllerLin+0.2, joystickLin+0.2);
	}
	else if(controllerLin < 0 && joystickLin < 0) {
		linear = std::max(controllerLin, joystickLin);
	}
	else{ linear = 0; }
	angular = std::max(angular,-1.0);
	joystickmsg.linear = std::min(linear, 1.0);
	joystickmsg.angular = std::min(angular,1.0);
}

//***************************************************
//
//      FUNCTION AND HELPERS TO SET WC VELOCITY
//
//      policy parameter reflects control mode (1 = proportional control; 2 = directional control)
//      id param represents who is setting speed (1 = controller joy; 2 = WC joy)
//
//***************************************************
double getScaledSpeed(double inputSpeed) {
	double limit = speed_limits[top_speed];
	return (inputSpeed * limit);
}

void setSpeedsHelper(PolarCoordinates polarCoords) {
	CartesianCoordinates cartesianCoords = polarToCartesianCoordinates(polarCoords);
	joystickmsg.linear = cartesianCoords.linear;
	joystickmsg.angular = cartesianCoords.angular;
}

void setSpeeds(double linear, double angular, int mode, int id) {

	PolarCoordinates polarCoordinates = cartesianToPolarCoordinates(linear, angular);

	if(mode == PROPCONTROL) {
		polarCoordinates.radius = getScaledSpeed(polarCoordinates.radius);
	}

	if(id == WCJOY && mode == DIRCONTROL && !controller_override) {
		polarCoordinates.radius = std::min(1.0, getScaledSpeed(polarCoordinates.radius));
	}

	if(id == WCJOY && mode == DIRCONTROL && controller_override) {
		polarCoordinates.radius = std::min(1.0, getScaledSpeed(polarCoordinates.radius));
		polarCoordinates.theta = ps3joystick.theta;
	}

	if(id == CONTROLLERJOY && mode == DIRCONTROL) {
		polarCoordinates.radius = std::min(1.0, getScaledSpeed(wcjoystick.radius));
		polarCoordinates.theta = ps3joystick.theta;
	}
	setSpeedsHelper(polarCoordinates);

}

//*******************************************************
//         CALLBACKS FOR BUTTON PRESSES
//*******************************************************
void toggleControlModeButtonCallback(std_msgs::Empty msg) {
	directional_control = !directional_control;
	joystickmsg.linear = 0;
	joystickmsg.angular = 0;
	publishDirectionalControlStatusMessage(directional_control);
}

void emergencyStopButtonCallback(std_msgs::Empty msg) {
	can_drive = false;
	joystickmsg.linear = 0;
	joystickmsg.angular = 0;
	publishEStopMessage(can_drive);
}

void resumeDrivingButtonCallback(std_msgs::Empty msg) {
	can_drive = true;
	publishEStopMessage(can_drive);
}

void increaseSpeedButtonCallback(std_msgs::Empty msg) {
	if (top_speed < 4) {
		top_speed += 1;
		publishLimitMessage(speed_limits[top_speed]);
	}
}

void decreaseSpeedButtonCallback(std_msgs::Empty msg) {
	if (top_speed > 0) {
		top_speed -= 1;
		publishLimitMessage(speed_limits[top_speed]);
	}
}

//UNCOMMENT BODY TO ENABLE PODRACING MODE FEATURE
void podRacingButtonCallback(std_msgs::Empty msg) {
	//pod_racing_mode = !pod_racing_mode;
}


//*******************************************************
//         CALLBACKS FOR JOYSTICKS
//*******************************************************
void controllerJoyCallback(JoyControl::joystickConstPtr joy_msg) {

	if(can_drive) {

		double linear = joy_msg->linear;
		double angular = joy_msg->angular;


		ps3joystick = cartesianToPolarCoordinates(linear, angular);


		controllerLin = linear;
		if(pod_racing_mode) {
			setPodRacingSpeed();
		}

		else if(linear != 0 || angular != 0 || controller_override == true) {
			controller_override = true;
			if(!directional_control) {
				setSpeeds(linear, angular, PROPCONTROL, CONTROLLERJOY);
			}
			else {
				setSpeeds(linear, angular, DIRCONTROL, CONTROLLERJOY);
			}
			//turn off override when controller joy no longer active
			if(linear == 0 && angular == 0) {
				controller_override = false;
			}
		}
	}

}

void wcJoyCallback(const sensor_msgs::Joy::ConstPtr& joyIn) {
	if(can_drive) {

		double linear = joyIn->axes[PS3_MOVE_AXIS_STICK_UPWARDS];
		double angular = joyIn->axes[PS3_MOVE_AXIS_STICK_LEFTWARDS];


        ///Lili feed substitude the linear and angular velocity with the wc_Joy
        //double linear = 0.0;
        //double angular = 0.0;

       // std::cout<<"linear is "<<linear<<std::endl;
       // std::cout<<"angular is "<<angular<<std::endl;


		wcjoystick = cartesianToPolarCoordinates(linear, angular);

		joystickLin = linear;
		if(pod_racing_mode) {
			setPodRacingSpeed();
		}

		else if(directional_control) {
			setSpeeds(linear, angular, DIRCONTROL, WCJOY);
		}
		else if(!controller_override){
			setSpeeds(linear, angular, PROPCONTROL, WCJOY);
		}

	}
}



//**************************************************
//              MAIN FUNCTION
//**************************************************
int main(int argc, char** argv) {

	ros::init(argc, argv, "inputListener");

	ros::NodeHandle n;

	// subscribe to topics
	ros::Subscriber toggle_control_mode_sub = n.subscribe("toggle_control_mode", 1, toggleControlModeButtonCallback);
	ros::Subscriber emergency_stop_sub = n.subscribe("emergency_stop", 1, emergencyStopButtonCallback);
	ros::Subscriber resume_driving_sub = n.subscribe("resume_driving", 1, resumeDrivingButtonCallback);
	ros::Subscriber increase_speed_sub = n.subscribe("increase_speed", 1, increaseSpeedButtonCallback);
	ros::Subscriber decrease_speed_sub = n.subscribe("decrease_speed", 1, decreaseSpeedButtonCallback);
	//ros::Subscriber controllerjoy_sub = n.subscribe("controller_joy", 1, controllerJoyCallback);
	//ros::Subscriber wcjoy_sub = n.subscribe("/wc_joy", 1, wcJoyCallback);
	ros::Subscriber podracing_sub = n.subscribe("toggle_pod_racing", 1, podRacingButtonCallback);

	//publish joystick output messages
	joy_output_pub.reset(new ros::Publisher);
	*joy_output_pub = n.advertise<JoyControl::joystick> ("/joy_output", 1);

    // publish limit messages
    limit_pub.reset(new ros::Publisher);
    *limit_pub = n.advertise<std_msgs::Int32> ("/limit", 1);

    // publish directionalControl status messages
    dirCtrl_pub.reset(new ros::Publisher);
    *dirCtrl_pub = n.advertise<std_msgs::Bool> ("/dirCtrl", 1);

    // publish Emergency Stop status messages
    eStop_pub.reset(new ros::Publisher);
    *eStop_pub = n.advertise<std_msgs::Bool> ("/eStop", 1);


	//ros::Rate r(ROS_SPIN_RATE); // 25 hz
 /*

	while (ros::ok()) {
		//send chair velocities to CanESDSenderJoystick
		publishJoyOutputMessages();

		ros::spinOnce();
		r.sleep();
	}
	*/
    double publish_period=0.0625;
	ros::Rate r(16); //16 hz


	std::string data_loc="/home/lci/workspace/wheelchair_files/seq1.txt";
	//predictedJoystick.txt";
    readData read_data(data_loc);

	double distance=0;

    while(ros::ok())
    {

        for(int i=0; i<read_data.lr_joy.size(); i++)
        {
          // joystickmsg.linear = read_data.fb_joy[i];
           // joystickmsg.angular= read_data.lr_joy[i];

            double linear= read_data.fb_joy[i];
            double angular= read_data.lr_joy[i];//*32767;
            ROS_INFO("the linear velocity is %lf", linear);

            cartesianToPolarCoordinates(linear, angular);
            setSpeeds(linear, angular, DIRCONTROL, WCJOY);

            publishJoyOutputMessages();
            ros::spinOnce();
            r.sleep();
        }
            break;
    }

    std::cout<<" The total distance travelled "<<distance<<std::endl;

    ros::Subscriber controllerjoy_sub = n.subscribe("controller_joy", 1, controllerJoyCallback);
	ros::Subscriber wcjoy_sub = n.subscribe("/wc_joy", 1, wcJoyCallback);

    while(ros::ok())
    {

        publishJoyOutputMessages();
        ros::spinOnce();
        r.sleep();
    }


}
