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

    std::ofstream fout1("/home/lci/workspace/wheelchair_files/seq1_copy1.txt");
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
        fb_joy.push_back(allDataPointsVec[i][1]);
        lr_joy.push_back(allDataPointsVec[i][2]);
        linear_vel.push_back(allDataPointsVec[i][3]);
        angular_vel.push_back(allDataPointsVec[i][4]);


    }

}

int main()
{


	std::string data_loc="/home/lci/workspace/wheelchair_files/seq1.txt";
	std::ofstream fout("/home/lci/workspace/wheelchair_files/seq1_copy.txt");
	//predictedJoystick.txt";
    readData read_data(data_loc);


        for(int i=0; i<read_data.lr_joy.size(); i++)
        {
          // joystickmsg.linear = read_data.fb_joy[i];
           // joystickmsg.angular= read_data.lr_joy[i];

            double linear= read_data.fb_joy[i];
            double angular= read_data.lr_joy[i];

             //cout<<"linear velocity is "<<linear<<endl;
            fout<<linear<<std::endl;
        }



}


