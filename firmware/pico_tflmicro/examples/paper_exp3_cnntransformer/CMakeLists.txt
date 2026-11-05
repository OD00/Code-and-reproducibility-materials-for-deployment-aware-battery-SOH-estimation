add_executable(paper_exp3_cnntransformer
    main.cpp
    main_functions.cpp
    model_data.cc
    test_fixture.cc
)
target_link_libraries(paper_exp3_cnntransformer
    pico-tflmicro
    pico-tflmicro_test
    pico_stdlib
    hardware_clocks
    hardware_timer
)
pico_enable_stdio_usb(paper_exp3_cnntransformer 1)
pico_enable_stdio_uart(paper_exp3_cnntransformer 0)
pico_add_extra_outputs(paper_exp3_cnntransformer)
