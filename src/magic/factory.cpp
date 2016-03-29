#include "magic/factory.hpp"

std::vector<AutoCleanable*> AutoCleanable::_singletons;
AutoCleanable AutoCleanable::_instance;
