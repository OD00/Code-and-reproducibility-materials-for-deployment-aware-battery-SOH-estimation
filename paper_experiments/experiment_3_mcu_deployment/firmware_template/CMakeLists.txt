add_executable(paper_experiment_3
    main.cpp
    main_functions.cpp
    model_data.cc
    test_fixture.cc
)
target_link_libraries(paper_experiment_3
    pico-tflmicro
    pico-tflmicro_test
    pico_stdlib
    hardware_timer
)
pico_enable_stdio_usb(paper_experiment_3 1)
pico_enable_stdio_uart(paper_experiment_3 0)
pico_add_extra_outputs(paper_experiment_3)
