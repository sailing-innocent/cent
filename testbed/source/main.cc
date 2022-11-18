#include <iostream>

#include <args/args.hxx>

int main(int argc, char** argv)
{
    args::ArgumentParser parser{
        "testbed\n",
    };
    args::HelpFlag help_flag{
        parser,
        "HELP",
        "Display this help menu",
        {'h', "help"}
    };
    
    try
    {
        parser.ParseCLI(argc, argv);
    }
    catch(const args::Help&)
    {
        std::cout << parser;
    }
    
#ifdef ENABLE_GUI
    std::cout << "The GUI is defined" << std::endl;
#endif 
    return 0;
}