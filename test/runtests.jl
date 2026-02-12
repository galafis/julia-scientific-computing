using Test

@testset "julia-scientific-computing Tests" begin
    @testset "Project Structure" begin
        @test isfile("README.md")
        @test isfile("LICENSE")
    end
    
    @testset "Basic Functionality" begin
        @test 2 + 2 == 4
    end
end
