cmake_minimum_required(VERSION 3.18)
project(sna_cpp VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2 /EHsc /arch:AVX2")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} /O2 /EHsc /arch:AVX2")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /O2 /EHsc /arch:AVX2")

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED)

pybind11_add_module(core_cpp
    binding.cpp
    neuron_izhikevich.cpp
    dendrite_compartment.cpp
    stdp_engine.cpp
    neuron_population.cpp
    innate_circuits.cpp
    global_workspace.cpp
    spa_neural_mapper.cpp
    cortical_brain.cpp
    language_layer.cpp
    predictive_coding_layer.cpp
    swr_engine.cpp
    replay_engine.cpp
    cortical_reactivation.cpp
    memory_recall.cpp
    sleep_wake_scheduler.cpp
    self_perception_network.cpp
    world_model.cpp
    synapse_buffer.cpp
    action_selector.cpp
    meta_cog.cpp
    intrinsic_value.cpp
    state_word_mapper.cpp
    narrative_self.cpp
    language_sequence.cpp
    autobiographical_memory.cpp
    counterfactual_engine.cpp
    qualia_layer.cpp
    intrinsic_motivation_engine.cpp
    metacognition_monitor.cpp
    spontaneous_thinker.cpp
    semantic_grounding.cpp
    temporal_depth.cpp
    theory_of_mind.cpp
    goal_generator.cpp
    emotion_system.cpp
    active_inference_engine.cpp
    social_interaction.cpp
    creative_generator.cpp
)

target_include_directories(core_cpp PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

target_compile_definitions(core_cpp PRIVATE NDEBUG _ITERATOR_DEBUG_LEVEL=0)

if(MSVC)
    target_compile_options(core_cpp PRIVATE /bigobj)
    set_target_properties(core_cpp PROPERTIES
        MSVC_RUNTIME_LIBRARY "MultiThreadedDLL"
    )
endif()

install(TARGETS core_cpp DESTINATION .)
