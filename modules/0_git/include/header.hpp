#pragma once
#include <string>

namespace myspace {
    class A { };
    std::string func(A& a);
};

class A;
std::string func(A& a);
