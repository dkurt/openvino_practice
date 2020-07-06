#pragma once
#include <string>



namespace myspace {
    class A { };
    std::string func(A& ab);
};

class A;
std::string func(A& ab);
