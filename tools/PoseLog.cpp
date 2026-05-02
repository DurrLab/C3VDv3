/***********************************************************************************/
/*
 *	File name:	PoseLog.cpp
 *
 *	Author:     Taylor Bobrow, Johns Hopkins University (2023)
 * 
 */

#include "PoseLog.h"

PoseLog::PoseLog( const std::string &filename)
{
    std::ifstream file;

    file.open(filename.c_str());

    if (!file)
        throw std::runtime_error("Error: could not open pose file " + filename );

    const float FPS = 29.97f;
    int frameIndex = 0;

    while (!file.eof())
    {
        std::string line;

        std::getline(file, line);

        if(file.eof())
            break;

        if(line.empty())
            continue;

        float time;

        glm::mat4 T(1.0);

        /* Parse timestamp plus pose in column-major order:
           time,c0r0,c0r1,c0r2,c0r3,c1r0,...,c3r3 */
        int n = sscanf(line.c_str(), "%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f",
                        &time,
                        &T[0][0], &T[0][1], &T[0][2], &T[0][3],
                        &T[1][0], &T[1][1], &T[1][2], &T[1][3],
                        &T[2][0], &T[2][1], &T[2][2], &T[2][3],
                        &T[3][0], &T[3][1], &T[3][2], &T[3][3]);

        if(n == 17)
        {
            trajectory[time] = T;
            frameIndex++;
            continue;
        }

        /* Parse pose in column-major order without an explicit timestamp.
           The timestamp is inferred from line order at 29.97 FPS. */
        int m = sscanf(line.c_str(), "%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f",
                        &T[0][0], &T[0][1], &T[0][2], &T[0][3],
                        &T[1][0], &T[1][1], &T[1][2], &T[1][3],
                        &T[2][0], &T[2][1], &T[2][2], &T[2][3],
                        &T[3][0], &T[3][1], &T[3][2], &T[3][3]);

        if(m != 16)
            throw std::runtime_error( "Error: " + filename + " is incorrectly formatted (expected either 16 column-major pose values, or 17 values with leading timestamp)" );

        time = frameIndex / FPS;
        trajectory[time] = T;
        frameIndex++;
    }

    file.close();

    /* The file should contain at least one pose. */
    if(trajectory.size() < 1)
        throw std::runtime_error("Error: the file " + filename + " does not contain at least one pose" );

}

/* Returns a pose linearly interpolated pose at t = @timestamp. */
glm::mat4 PoseLog::getTransform(const float &timestamp)
{
    /* Check that the requested time is within bounds. */
    if( !(timestamp >= getBeginTime()  &&  timestamp <= getEndTime()) )
        throw std::runtime_error( "The requsted pose time is not within bounds." );

    std::map<float, glm::mat4>::const_iterator it1 = trajectory.lower_bound(timestamp);

    if(it1 != trajectory.end() && it1->first == timestamp)
        return it1->second;

    std::map<float, glm::mat4>::const_iterator it0 = std::prev(it1);

    float t0 = it0->first;
    glm::mat4 v0 = it0->second;
    float t1 = it1->first;
    glm::mat4 v1 = it1->second;

    /* Return a weighted linear interpolation of the two closest poses. */
    float w = (timestamp-t0)/(t1-t0);
    glm::quat r0 = glm::quat_cast(v0);
    glm::quat r1 = glm::quat_cast(v1);
    glm::quat rw = glm::slerp(r0,r1,w);
    
    glm::vec4 p0 = glm::column(v0,3);
    glm::vec4 p1 = glm::column(v1,3);
    glm::vec4 pw = glm::mix(p0,p1,w);

    glm::mat4 vw = glm::mat4_cast(rw);
    vw = glm::column(vw,3,pw);

    return vw;
}

float PoseLog::getBeginTime()
{
    std::map<float, glm::mat4>::const_iterator it = trajectory.begin();
    return it->first;
}

float PoseLog::getEndTime()
{
    std::map<float, glm::mat4>::const_iterator it = std::prev(trajectory.end());
    return it->first;
}
