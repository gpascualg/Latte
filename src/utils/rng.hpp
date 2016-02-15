#pragma once

#include "pcg32.h"

class Rng
{
public:
    static pcg32* instance()
    {
        if (!_instance)
        {
            _instance = new Rng();
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
