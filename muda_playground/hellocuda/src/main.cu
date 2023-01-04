#include <muda/muda.h>

int main()
{
    muda::launch(1,1)
        .apply(
            [] __device__()
            {
                printf("hello muda!\n");
            }
        ).wait();
    return 0;
}
