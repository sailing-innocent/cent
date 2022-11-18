#include <iostream>
#include <args/args.hxx>

#include <testbed/testbed.h>

using namespace testbed;

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
    args::Flag no_gui_flag{
        parser,
        "NO_GUI",
        "Disable the GUI",
        {"no-gui"}
    };
    
    try
    {
        parser.ParseCLI(argc, argv);
    }
    catch(const args::Help&)
    {
        std::cout << parser;
    }
    catch (const args::ParseError& e)
    {
        std::cerr << e.what() << std::endl;
        std::cerr << parser;
        return 1;
    }
    
    if (no_gui_flag) {
        std::cout << "The GUI is disabled" << std::endl;
    }

    bool gui = !no_gui_flag;
#ifndef ENABLE_GUI
    std::cout << "The GUI is not enabled" << std::endl;
    gui = false;
#endif 
    
    try {
        ITestbedMode mode{RaytraceMesh};
        Testbed testbed{mode};
        if (gui) {
            testbed.init_window(800, 600);
        }
        while (testbed.frame()) {
            // if no gui, log instead
        }
    } catch (const std::exception& e) {
        std::cerr << "EXCEPTION" << std::endl;
        return 1;
    }

    return 0;
}