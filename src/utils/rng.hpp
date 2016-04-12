#pragma once

#include "pcg32.h"
#include <time.h>


class Rng
{
public:
    static pcg32* instance()
    {
        if (!_instance)
        {
            _instance = new Rng();
	    _instance->_rng.seed(time(NULL));
        }
        
        return &_instance->_rng;
    }
    
private:
    Rng() {}
    
private:
    pcg32 _rng;
    static Rng* _instance;    
};

inline pcg32* rng()
{
    return Rng::instance();
}
