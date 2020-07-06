#include <string>

#include <gtest/gtest.h>

#include "header.hpp"


TEST(git, say_hello) {
    myspace::A a;
    EXPECT_EQ(myspace::func(a), "Hello, Nizhny!");
    EXPECT_EQ(func(a), "Hello, Nizhny!");
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
