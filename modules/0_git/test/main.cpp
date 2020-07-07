#include <string>

#include <gtest/gtest.h>

#include "header.hpp"

TEST(git, say_hello) {
    A a;
    myspace::A b;
    EXPECT_EQ(myspace::func(b), "Hello, Nizhny!");
    EXPECT_EQ(func(a), "Hello, Nizhny!");
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
