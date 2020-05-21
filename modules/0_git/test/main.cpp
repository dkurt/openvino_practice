#include <string>

#include <gtest/gtest.h>

namespace myspace {
    class A { };
    std::string func(A& a) {
        return "Hello, World!";
    }
};

class A;
std::string func(A& a) {
    return "Hello, Nizhny!";
}

TEST(git, say_hello) {
    myspace::A a;
    EXPECT_EQ(myspace::func(a), "Hello, World!");
    EXPECT_EQ(func(a), "Hello, Nizhny!");
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
