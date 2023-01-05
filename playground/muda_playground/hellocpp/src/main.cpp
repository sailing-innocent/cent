#include <iostream>
#include "muda/muda.h"

using namespace std;

int main(int argc, char** argv)
{
    cout << "hello world!" << endl;

    muda::launch(1,1)
        .apply(
            [] __device__()
            {
                muda::print("hello muda!\n");
            }
        ).wait();
    return 0;
}
