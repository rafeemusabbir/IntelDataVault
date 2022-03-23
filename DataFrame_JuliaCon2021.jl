### A Pluto.jl notebook ###
# v0.16.2

using Markdown
using InteractiveUtils

# ╔═╡ 4c2c0635-9a57-4816-91b4-ccc5bdd39981
begin
    import Pkg 
    Pkg.activate(mktempdir()) 
    Pkg.add([
		Pkg.PackageSpec(name="DataFrames"),  
		Pkg.PackageSpec(name="Statistics"),
		Pkg.PackageSpec(name="Random"), 
		Pkg.PackageSpec(name="GLM"),
		Pkg.PackageSpec(name="StatsPlots"),
		Pkg.PackageSpec(name="StatsBase"),
        Pkg.PackageSpec(name="Plots"),
        Pkg.PackageSpec(name="PlutoUI"),
		Pkg.PackageSpec(name="CSV"),
        Pkg.PackageSpec(name="LaTeXStrings"),
        Pkg.PackageSpec(name="Bootstrap"),
        Pkg.PackageSpec(name="Chain"),
		Pkg.PackageSpec(name="CategoricalArrays")
    ])
    using PlutoUI, LaTeXStrings
	using Random, Bootstrap
	using Statistics, StatsBase
	using CSV, CategoricalArrays, Chain
	using DataFrames
	using Plots, StatsPlots
	using GLM
	import Downloads
end

# ╔═╡ 8facaea4-aa0c-4973-9c2d-a3e7b67c5799
md"""
A little bit about myself:

* **Sarder Rafee Musabbir**, PhD Candidate 🌐 University of California davis
* Autonomous Vehicle, Dynamic Games, and Data Science Enthusiast trained in **Traffic Operation & Control**, **Statistics**, **Game Theory**, **Machine Learning** 🚀
* Make contents on [**Optimal Control**](), [**Data Science**](), [**Scientific Machine Learning**](), [**Stochastic Optimization**]() 🚀
* You can find me on [Github](https://github.com/rafeemusabbir) or on [LinkedIn](https://www.linkedin.com/in/sarder-rafee-musabbir-95446a73/)
"""

# ╔═╡ 0d28a09b-339a-4f87-9f5f-9c2579ceeea5
md"""
# Some Statistical Concepts

In this tutorial we will use the following statistical concepts:

* [Confidence Intervals](https://en.wikipedia.org/wiki/Confidence_interval)
* [Density Estimators](https://en.wikipedia.org/wiki/Density_estimation)
* [Probit Model](https://en.wikipedia.org/wiki/Probit_model)
* [Bootstrapping](https://en.wikipedia.org/wiki/Bootstrapping_(statistics))

If you want to follow the material smoothly it is recommended that you review them before continuing. The objective of the tutorial is to give you an overview of the ecosystem. This means that I will show many things but will not discuss them deeply.
"""

# ╔═╡ c6d794ce-4d5f-4ed9-90d8-afdc0cadc4aa
md"""
The tutorial was developed and tested under Julia 1.6.1 and should be run with Project.toml and Manifest.toml files present in the working directory of the notebook (the files are provided in the https://github.com/bkamins/Julia-DataFrames-Tutorial repository).
"""

# ╔═╡ bfb8a2f0-1f9b-45f4-b1a4-00d6476ba2ae
md"""
# Data Pre-Processing
"""

# ╔═╡ 9f43399a-4b08-490f-8199-d535ff7d8ebd
Downloads.download("https://vincentarelbundock.github.io/Rdatasets/csv/Ecdat/Participation.csv", "participation.csv")

# ╔═╡ d9c3978f-407c-47a1-af78-ad3c1a770ede
readlines("participation.csv")

# ╔═╡ 83692eb9-b3e3-478a-8e6e-f266c63319a3
df_raw = CSV.read("participation.csv", DataFrame)

# ╔═╡ ae4b50a8-dd21-40f7-a813-645c979458f7
describe(df_raw)

# ╔═╡ 549194c1-2092-4714-80a9-e6d1029179dd
md"""
Now transform the data set. We use the `select` function and perform the following transformations:
* recode `:lfp` variable from text to binary;
* add square of `:age` column;
* change `:foreign` column to be categorical;
* all other columns are left as they are.

The general syntax of column transformation options is:
```julia
source columns => transformation => target columns name
```
(the `transformation` and `target columns name` can be dropped)

Also note the `ByRow` wrapper which tells `select` to perform the operation row-wise (by default operations are performed on whole columns).

In [this post](https://bkamins.github.io/julialang/2020/12/24/minilanguage.html) you can find an explanation of the most typical cases of supported transformation specifications.
"""

# ╔═╡ d6fc8d9f-d1a8-4cc6-b52e-b62482d0a9a9
# df = select(df_raw,
#             :lfp => (x -> recode(x, "yes" => 1, "no" => 0)) => :lfp,
#             :lnnlinc,
#             :age,
#             :age => ByRow(x -> x^2) => :age²,
#             Between(:educ, :noc),
#             :foreign => categorical => :foreign)

# ╔═╡ 636d22e9-1d3f-48fa-bc2b-b542efb5b6de
md"""
If we want to leave column names unchanged we can use renamecols=false keyword argument, and in this case the target column name can be dropped:
"""

# ╔═╡ 01835bd4-e084-4ed4-a3c0-c154c57beaca
df = select(df_raw,
            :lfp => x -> recode(x, "yes" => 1, "no" => 0),
            :lnnlinc,
            :age,
            :age => ByRow(x -> x^2) => :age²,
            Between(:educ, :noc),
            :foreign => categorical,
            renamecols=false)

# ╔═╡ 91b63aa8-9218-41dd-88b8-4c6e24469779
describe(df)

# ╔═╡ 00a13dfc-4ee5-448a-b19c-d4d404737c54
md"""
# Exploratory Data Analysis
"""

# ╔═╡ eff83cca-a3f8-48cb-a032-51d78e017eb6
md"""
We want to compute the mean of numeric columns by `:lfp` to initialy check in what direction they influence the target.

In the example we use the following:
* the `@chain` macro from the [Chain.jl](https://github.com/jkrumbiegel/Chain.jl) package; it allows for convenient piping of operations;
* the `groupby` function that adds the key column to the data frame (groups the data frame by the passed column)
* the `combine` function that combines the rows of a data frame by some function
"""

# ╔═╡ 4041d1a7-c5c6-413d-affd-3d5b15573c47
@chain df begin
    groupby(:lfp)
    combine([:lnnlinc, :age, :educ, :nyc, :noc] .=> mean)
end

# ╔═╡ 74df5239-7893-455c-80d3-b2c79b48742a
[:lnnlinc, :age, :educ, :nyc, :noc] .=> mean

# ╔═╡ 46aa280e-4124-4cfb-992f-69039c9e496a
md"""
is a convenient way to specify multiple similar transformations using the broadcasting syntax provided by Julia.

You can find more examples of this pattern at work [here](https://bkamins.github.io/julialang/2021/07/09/multicol.html).
"""

# ╔═╡ 52c2223e-43e2-4c70-83f2-af5014e87da1
md"""
If we did not want to list all the columns manually we could have written (note that `:lfp` was included as it is binary):
"""

# ╔═╡ 93a7bab0-54e5-44fc-85ab-ef8cebe653ef
@chain df begin
    groupby(:lfp)
    combine(names(df, Real) .=> mean)
end

# ╔═╡ 61cc0d0c-b16e-4ff3-814a-faf29251da08
md"""
Now handle the categorical variable `:foreign`.

Note that this time we pass just `nrow` to combine, which has a special treatement and returns the number of rows in each group.
"""

# ╔═╡ fd2489e2-2b36-49bc-9730-21071bca33e8
@chain df begin
    groupby([:lfp, :foreign])
    combine(nrow)
end

# ╔═╡ a0113989-06b6-4846-8855-8fbff73b38ac
md"""
If we wanted to create a cross-tabulation of the data we can put:
* the `:lfp` variable as rows,
* the `:foreign` variable as columns,
* the `:nrow` variable as values,

using the `unstack` function.

More advanced examples of reshaping of data frames are presented [here](https://bkamins.github.io/julialang/2021/05/28/pivot.html).
"""

# ╔═╡ b0c5d8ff-f768-415c-98f3-6048c417d047
@chain df begin
    groupby([:lfp, :foreign])
    combine(nrow)
    unstack(:lfp, :foreign, :nrow)
end

# ╔═╡ 77234766-0a76-4544-80f9-1f58ffa67ce1
md"""
Finally let us add another step to our `@chain`, which will create a fraction of `:yes` answers in each group.

Note that in this example we show how to pass more than one column to a transformation function.
"""

# ╔═╡ 7091aa87-8268-4496-9f81-6b473d0f2010
@chain df begin
    groupby([:lfp, :foreign])
    combine(nrow)
    unstack(:lfp, :foreign, :nrow)
    select(:lfp, [:no, :yes] => ByRow((x, y) -> y / (x + y)) => :foreign_yes)
end

# ╔═╡ 0fa6d5d7-c0b5-454a-8872-302cd63634b9
md"""
An observant reader will notice that we could have done it in one step like this (the benefit of being verbose was that we have learned more features of DataFrames.jl):
"""

# ╔═╡ 9bcad67a-f78f-4a0f-acad-3e14f70752d4
@chain df begin
    groupby(:lfp)
    combine(:foreign => (x -> mean(x .== "yes")) => :foreign_yes)
end

# ╔═╡ 5f2c4609-77f4-485a-bad0-849b7826c1f8
md"""
The `GroupedDataFrame` that is created by the `groupby` function can be a useful object to work with on its own.
"""

# ╔═╡ 29075999-3d7a-4f13-80a4-39fd3b364e7f
gd = groupby(df, :lfp)

# ╔═╡ 96b1fe70-d77e-404c-85d1-507770430f72
md"""
Note that you can conveniently index into it to get the groups. \

First we use the standard indexing syntax:
"""

# ╔═╡ 528f4f6f-9c2e-45b0-87a0-154f8e0baf3e
gd[1]

# ╔═╡ cd65c724-63f6-4088-889b-c6db0b9a1344
md"""
Now we we use a special indexing syntax that efficiently selects groups by their value (not position):
"""

# ╔═╡ b62c533e-7630-484d-88ec-fb377eaadd3a
gd[(lfp=0,)]

# ╔═╡ 70432aed-fedc-4add-af6d-bc4dcd7abcb0
gd[Dict(:lfp => 0)]

# ╔═╡ dbf0ad5e-276f-4369-98be-471878d7480b
gd[(0,)]

# ╔═╡ 1075d6c1-5d1a-45e9-b045-1c09bbc1a694
md"""
Before we move forward let us investigate why we have added a square of `:age` to our data frame.

For this we use density plot from the [StatsPlots.jl](https://github.com/JuliaPlots/StatsPlots.jl) package. This package contains statistical recepies that extend the [Plots.jl](https://github.com/JuliaPlots/Plots.jl) functionality.
"""

# ╔═╡ 1c646bf2-9c94-4a99-99c0-9def436d504a
@df df density(:age, group=:lfp)

# ╔═╡ 65145299-08c2-4b24-a6ec-21e084b54c85
md"""
# Building a Predictive Model
"""

# ╔═╡ e63c5bda-78bb-401c-a61d-84aa0e7d84f0
md"""
Now we are ready to create the probit model using the [GLM.jl](https://github.com/JuliaStats/GLM.jl) package:
"""

# ╔═╡ 26fb9ddf-54bc-4fff-be3e-d9a28a7464a7
probit = glm(@formula(lfp ~ lnnlinc + age + age² + educ + nyc + noc + foreign),
             df, Binomial(), ProbitLink())

# ╔═╡ 9ea418cb-faff-43ef-9264-2bb6265d3713
md"""
In the example above we have entered the `@formula` by hand. However, it can be also programmatically generated like this:
"""

# ╔═╡ 92206350-69a5-4f6e-825d-a4d326e7e8fd
lprobit = glm(Term(:lfp) ~ sum(Term.(propertynames(df)[2:end])),
             df, Binomial(), ProbitLink())

# ╔═╡ 9b392369-cf35-4c03-a044-7909c6fd2283
md"""
Note the following:
"""

# ╔═╡ 390b320e-2804-420e-8061-a7baad418bad
Term(:lfp) ~ sum(Term.(propertynames(df)[2:end]))

# ╔═╡ 3183200e-a9b5-4102-978e-3c001344c890
@formula(lfp ~ lnnlinc + age + age² + educ + nyc + noc + foreign)

# ╔═╡ f2c4b033-60e8-47f2-8a9d-75b188039b54
md"""
Finally observe that `@formula` is powerful enough to automatically do the computation of the square of `:age`:
"""

# ╔═╡ b4ba4a32-7639-4351-9aa1-e59463026546
lnprobit = glm(@formula(lfp ~ lnnlinc + age + age^2 + educ + nyc + noc + foreign),
             df, Binomial(), ProbitLink())

# ╔═╡ dbbe9341-fe6b-4bcc-a1dc-b2eb1d5f74bc
md"""
We check the formula again:
"""

# ╔═╡ b782024a-cd83-4e7e-a53e-b78abc090f91
@formula(lfp ~ lnnlinc + age + age^2 + educ + nyc + noc + foreign)

# ╔═╡ 8bf59bed-0dce-4f78-be29-ed253de17f37
md"""
Next we prepare a new data frame in which we will check how the prediction of our model changes as we modify `:age` while keeping all other variables constant:
"""

# ╔═╡ 833dc0f7-8322-4720-9b67-e2a6b49ea9b6
df_pred = DataFrame(lnnlinc=10.0, age= 2.0:0.01:6.2, educ = 9, nyc = 0, noc = 1, foreign = "yes")

# ╔═╡ 943054c2-652f-4679-b930-f62299669cc7
md"""
We make a prediction along with its confidence interval:
"""

# ╔═╡ b5e9d192-97e6-4d8d-8af3-fa92a335f751
probit_pred = predict(lnprobit, df_pred, interval=:confidence)

# ╔═╡ f020f614-b58a-400e-ad9e-12706b700bb9
md"""
Now we plot it. Note that we use `Matrix` constructor to create a matrix out of the data frame easily:
"""

# ╔═╡ 66f6c54a-2056-4312-ab81-a0e914d97991
plot(df_pred.age, Matrix(probit_pred), labels=[L"lfp" L"lower" L"upper"],
     xlabel=L"age", ylabel=L"Pr(lfp=1)")

# ╔═╡ 64cdf513-ba9d-4661-b47d-c78e2cb7fe47
md"""
# Advanced Functionalities: Bootstrapping
"""

# ╔═╡ 28722b31-ef74-4f32-b7af-0fc3a4788ada
md"""
Let us investigate the `probit` object again:
"""

# ╔═╡ e88687af-3f52-471e-b566-38138f5110f0
probit

# ╔═╡ f126fa16-13c0-4066-a7fa-e2d1a34e9666
md"""
We can see that we obtained parametric confidence intervals for the parameters. However, our sample was not very big, so we want to verify them using bootstrapping.

First we will do bootstrapping by hand, and next we will compare the results with what the [Bootstrap.jl](https://github.com/juliangehring/Bootstrap.jl) package produces.

As usual we will try to learn some new features of DataFrames.jl along the way.
"""

# ╔═╡ ff8b0889-177b-4ecd-b593-8ba340caf937
md"""
We start with a function that takes a data frame and:
1. creates one bootstrap sample of its contents
2. fits the probit model to the bootstrapped data
3. returns a `NamedTuple` with the computed coefficients
"""

# ╔═╡ 6a530502-f10c-4a4f-b398-1a066dfeb04f
function boot_sample(df)
    df_boot = df[rand(1:nrow(df), nrow(df)), :]
    probit_boot = glm(@formula(lfp ~ lnnlinc + age + age^2 + educ + nyc + noc + foreign),
                      df_boot, Binomial(), ProbitLink())
    return (; (Symbol.(coefnames(probit_boot)) .=> coef(probit_boot))...)
end

# ╔═╡ f55e2e87-4483-4884-b8d7-a2bd4de80418
md"""
We need to run the `boot_sample` function multiple times. Note that we store the results in the `coef_boot` data frame.
"""

# ╔═╡ 5594ca50-f831-4ae4-813c-ee7ea896abe8
function run_boot(df, reps)
    coef_boot = DataFrame()
    for _ in 1:reps
        push!(coef_boot, boot_sample(df))
    end
    return coef_boot
end

# ╔═╡ e50cfd31-f963-4690-8bd0-ebdaa30aa4cf
md"""
We seed the random number generator as we want comparable results to what Bootstrap.jl produces (I have made sure to sample rows for bootstrapping in the same way in our manual code).
"""

# ╔═╡ 35e2c3de-7ec0-45cc-916e-eaa5e0703f45
begin
Random.seed!(1234)
@time coef_boot = run_boot(df, 1000)
end

# ╔═╡ d1bd2e80-5d5d-4626-af59-825af04135a3
md"""
Using this data calculate the 95% confidence interval using the percentile bootstrap:
"""

# ╔═╡ 197e994a-7af2-46da-a643-116c47087db5
conf_boot = mapcols(x -> quantile(x, [0.025, 0.975]), coef_boot)

# ╔═╡ d1856fed-1ea6-43be-876e-44e73a6fb13a
md"""
Here are the parametric confidence intervals computed by GLM.jl. We will want to compare them against bootstrapping results:
"""

# ╔═╡ 17b44487-66c6-4810-b9b0-38d8c890db9a
confint(probit)

# ╔═╡ 6ed640a4-20b1-422a-8888-1c93203788bb
md"""
First we transform the above matrix and create a data frame using the same column names as in bootstrapping:
"""

# ╔═╡ 448c0078-b2d3-4597-b30b-367f329603e2
conf_param = DataFrame(permutedims(confint(probit)), names(conf_boot))

# ╔═╡ 493d0cce-2f99-471a-b791-4287965f13d1
md"""
and we `append!` it to our `conf_boot` data frame:
"""

# ╔═╡ fd46bfb8-91d9-4d49-8f98-48f493319f76
append!(conf_boot, conf_param)

# ╔═╡ 4f08fc6d-aa8e-4718-99e0-c8da07fe52dc
md"""
It is good to keep track of what each row of our data holds. Therefore we insert a new column to our data farme. As we want to put it in front we use the `insertcols!` function.
"""

# ╔═╡ 2f9fdaa9-9183-4cd1-85e4-816aaa4b51ca
insertcols!(conf_boot, 1, :statistic => ["boot lo", "boot hi", "parametric lo", "parametric hi"])

# ╔═╡ 53717c9c-475c-4374-a3fb-2f5459b2a5d3
md"""
Notice that data frame also can be transposed. However, we need to provide a column that we will use as column names in the target data frame (data frame objects must have column names):
"""

# ╔═╡ ce86e8af-fa5e-41da-9568-5379c0328b80
conf_boot_t = permutedims(conf_boot, :statistic)

# ╔═╡ f20ff624-d13f-4796-89d7-d13c7d87d4b9
md"""
Let us also add the estimates of the coefficients to the table:
"""

# ╔═╡ 7c9d1e81-2a5c-4b5a-a625-53cb46011eea
insertcols!(conf_boot_t, 2, :estimate => coef(probit))

# ╔═╡ d267d626-589a-4f64-9362-28653ba1503f
md"""
Now it is time for some more advanced stuff. We want to transform columns holding the ends of the confidence intervals (which are columns from 3 to 6) into absolute deviations from the estimate. Such a transformation will be useful for plotting:
"""

# ╔═╡ 6866d623-70d1-49b8-a179-b9040a8a2ded
select!(conf_boot_t, :statistic, :estimate, 3:6 .=> x -> abs.(x .- conf_boot_t.estimate), renamecols=false)

# ╔═╡ 9ee2a814-94b7-49ec-bcfc-29fe2767c7b3
md"""
I have promised you a plot, so let us not hesitate producing it:
"""

# ╔═╡ 022b95d8-c013-4f18-945a-6df5fe917e3e
begin
scatter(0.05 .+ (1:8), conf_boot_t.estimate,
        yerror=(conf_boot_t."boot lo", conf_boot_t."boot hi"),
        label="bootstrap",
        xticks=(1:8, conf_boot_t.statistic), xrotation=45)
scatter!(-0.05 .+ (1:8), conf_boot_t.estimate,
         yerror=(conf_boot_t."parametric lo", conf_boot_t."parametric hi"),
         label="parametric")
end

# ╔═╡ 86ebd244-1415-4acc-baaa-6d7a9ba9668f
md"""
As you can see both types of intervals are quite close in this case. \
Before we finish let us get a sample of how the same task could have been done using the Bootstrap.jl package. \
This time the function compiting the statistics does not have to perform sampling as this is handled by the Bootstrap.jl package:
"""

# ╔═╡ 0dc372ca-0896-46e3-815b-965235c4512c
function boot_probit(df_boot)
    probit_boot = glm(@formula(lfp ~ lnnlinc + age + age^2 + educ + nyc + noc + foreign),
                      df_boot, Binomial(), ProbitLink())
    return (; (Symbol.(coefnames(probit_boot)) .=> coef(probit_boot))...)
end

# ╔═╡ 4242c1fa-114b-4f14-a8ee-78fbbf472957
md"""
First we create a bootstrap sample:
"""

# ╔═╡ f55696eb-3658-4296-82d6-93ba51d77052
begin
Random.seed!(1234)
@time bs = bootstrap(boot_probit, df, BasicSampling(1000))
end

# ╔═╡ 1a6bdcb9-6858-4f99-84c3-1347f78ef772
md"""
and next we compute 95% percentile confidence intervals:
"""

# ╔═╡ 53754be3-0d75-4365-a924-8810725870b9
bs_ci = confint(bs, PercentileConfInt(0.95))

# ╔═╡ 3b5d6394-6211-47e6-9357-bc69b2471aca
md"""
Let us chceck if the result matches our manual invervals. First create a new column in our data frame containing tuples. This is a typical example of *nested* data structure, that is handled by DataFrames.jl without any problems.

Notice that I compute the deviations of lower and upper ends of confidence intervals from the estimate to match our earlier computations.
"""

# ╔═╡ 33b686f4-020d-471b-91e4-b32a0da1bed6
conf_boot_t.bootstrap = [(ci[1], ci[1] - ci[2], ci[3] - ci[1]) for ci in bs_ci]

# ╔═╡ 32feb06e-a7cb-4af9-9f6d-c66f089a1804
md"""
We could have inspected our data frame now:
"""

# ╔═╡ 36c15c2e-bb44-4d82-89bf-5b7d4bfb6da8
conf_boot_t

# ╔═╡ 3328cf9d-5382-4df0-8aff-271fa858c99d
md"""
But it is a bit inconvenient. \
First un-nest the `:bootstrap` column into three columns:
"""

# ╔═╡ 6fe94317-e4a5-42be-abc7-406735482c5c
select!(conf_boot_t, Not(:bootstrap), :bootstrap => ["estimate 2", "boot lo 2", "boot hi 2"])

# ╔═╡ 89cfa6e6-d0d1-4616-90a9-07e02efcb76a
md"""
Next reorder the columns using regular expressions.
"""

# ╔═╡ 443d1273-3762-4631-85f0-d29e130d5245
select(conf_boot_t, :statistic, r"estimate", r"lo", r"hi")

# ╔═╡ dfdba161-8775-42cf-8aa5-7c6129ca4620
md"""
More discussion of various column selectors supported can be found [here](https://bkamins.github.io/julialang/2021/02/06/colsel.html). \
Now we can more easily see that our manual computations produce exactly the same results as the Bootstrap.jl package. \
Before we finish sort our data frame by the `:estimate`:
"""

# ╔═╡ 7c63277d-a47d-4ab3-aca9-221a1091f610
sort(conf_boot_t, :estimate)

# ╔═╡ 6b50d54b-54d9-429c-94df-97f764c21559
md"""
Here is some more advanced example, where we sort the rows by the width of the confidence interval:
"""

# ╔═╡ 7d023b1c-4b49-4c7e-b755-5a785f1eac1b
conf_boot_t[sortperm(conf_boot_t."boot hi" + conf_boot_t."boot lo"), :]

# ╔═╡ b77d0c93-e215-41a3-99fc-ad367c6db153
md"""
More examples of sorting data frames can be found [here](https://bkamins.github.io/julialang/2021/03/12/sorting.html).
"""

# ╔═╡ 9432dfff-4b44-4236-8b5d-f80fddc69d3a
md"""
### Pluto Notebook Essentials
"""

# ╔═╡ 7aab7828-9a01-4393-a5bf-ee348b763112
TableOfContents(title="📚 DataFrame JuliaCon 2021", aside=true) 

# ╔═╡ Cell order:
# ╟─8facaea4-aa0c-4973-9c2d-a3e7b67c5799
# ╟─0d28a09b-339a-4f87-9f5f-9c2579ceeea5
# ╟─c6d794ce-4d5f-4ed9-90d8-afdc0cadc4aa
# ╟─bfb8a2f0-1f9b-45f4-b1a4-00d6476ba2ae
# ╠═9f43399a-4b08-490f-8199-d535ff7d8ebd
# ╠═d9c3978f-407c-47a1-af78-ad3c1a770ede
# ╠═83692eb9-b3e3-478a-8e6e-f266c63319a3
# ╠═ae4b50a8-dd21-40f7-a813-645c979458f7
# ╟─549194c1-2092-4714-80a9-e6d1029179dd
# ╠═d6fc8d9f-d1a8-4cc6-b52e-b62482d0a9a9
# ╟─636d22e9-1d3f-48fa-bc2b-b542efb5b6de
# ╠═01835bd4-e084-4ed4-a3c0-c154c57beaca
# ╠═91b63aa8-9218-41dd-88b8-4c6e24469779
# ╟─00a13dfc-4ee5-448a-b19c-d4d404737c54
# ╟─eff83cca-a3f8-48cb-a032-51d78e017eb6
# ╠═4041d1a7-c5c6-413d-affd-3d5b15573c47
# ╠═74df5239-7893-455c-80d3-b2c79b48742a
# ╟─46aa280e-4124-4cfb-992f-69039c9e496a
# ╟─52c2223e-43e2-4c70-83f2-af5014e87da1
# ╠═93a7bab0-54e5-44fc-85ab-ef8cebe653ef
# ╟─61cc0d0c-b16e-4ff3-814a-faf29251da08
# ╠═fd2489e2-2b36-49bc-9730-21071bca33e8
# ╟─a0113989-06b6-4846-8855-8fbff73b38ac
# ╠═b0c5d8ff-f768-415c-98f3-6048c417d047
# ╟─77234766-0a76-4544-80f9-1f58ffa67ce1
# ╠═7091aa87-8268-4496-9f81-6b473d0f2010
# ╟─0fa6d5d7-c0b5-454a-8872-302cd63634b9
# ╠═9bcad67a-f78f-4a0f-acad-3e14f70752d4
# ╟─5f2c4609-77f4-485a-bad0-849b7826c1f8
# ╠═29075999-3d7a-4f13-80a4-39fd3b364e7f
# ╟─96b1fe70-d77e-404c-85d1-507770430f72
# ╠═528f4f6f-9c2e-45b0-87a0-154f8e0baf3e
# ╟─cd65c724-63f6-4088-889b-c6db0b9a1344
# ╠═b62c533e-7630-484d-88ec-fb377eaadd3a
# ╠═70432aed-fedc-4add-af6d-bc4dcd7abcb0
# ╠═dbf0ad5e-276f-4369-98be-471878d7480b
# ╟─1075d6c1-5d1a-45e9-b045-1c09bbc1a694
# ╠═1c646bf2-9c94-4a99-99c0-9def436d504a
# ╟─65145299-08c2-4b24-a6ec-21e084b54c85
# ╟─e63c5bda-78bb-401c-a61d-84aa0e7d84f0
# ╠═26fb9ddf-54bc-4fff-be3e-d9a28a7464a7
# ╟─9ea418cb-faff-43ef-9264-2bb6265d3713
# ╠═92206350-69a5-4f6e-825d-a4d326e7e8fd
# ╟─9b392369-cf35-4c03-a044-7909c6fd2283
# ╠═390b320e-2804-420e-8061-a7baad418bad
# ╠═3183200e-a9b5-4102-978e-3c001344c890
# ╟─f2c4b033-60e8-47f2-8a9d-75b188039b54
# ╠═b4ba4a32-7639-4351-9aa1-e59463026546
# ╟─dbbe9341-fe6b-4bcc-a1dc-b2eb1d5f74bc
# ╠═b782024a-cd83-4e7e-a53e-b78abc090f91
# ╟─8bf59bed-0dce-4f78-be29-ed253de17f37
# ╠═833dc0f7-8322-4720-9b67-e2a6b49ea9b6
# ╟─943054c2-652f-4679-b930-f62299669cc7
# ╠═b5e9d192-97e6-4d8d-8af3-fa92a335f751
# ╟─f020f614-b58a-400e-ad9e-12706b700bb9
# ╠═66f6c54a-2056-4312-ab81-a0e914d97991
# ╟─64cdf513-ba9d-4661-b47d-c78e2cb7fe47
# ╟─28722b31-ef74-4f32-b7af-0fc3a4788ada
# ╠═e88687af-3f52-471e-b566-38138f5110f0
# ╟─f126fa16-13c0-4066-a7fa-e2d1a34e9666
# ╟─ff8b0889-177b-4ecd-b593-8ba340caf937
# ╠═6a530502-f10c-4a4f-b398-1a066dfeb04f
# ╟─f55e2e87-4483-4884-b8d7-a2bd4de80418
# ╠═5594ca50-f831-4ae4-813c-ee7ea896abe8
# ╟─e50cfd31-f963-4690-8bd0-ebdaa30aa4cf
# ╠═35e2c3de-7ec0-45cc-916e-eaa5e0703f45
# ╟─d1bd2e80-5d5d-4626-af59-825af04135a3
# ╠═197e994a-7af2-46da-a643-116c47087db5
# ╟─d1856fed-1ea6-43be-876e-44e73a6fb13a
# ╠═17b44487-66c6-4810-b9b0-38d8c890db9a
# ╟─6ed640a4-20b1-422a-8888-1c93203788bb
# ╠═448c0078-b2d3-4597-b30b-367f329603e2
# ╟─493d0cce-2f99-471a-b791-4287965f13d1
# ╠═fd46bfb8-91d9-4d49-8f98-48f493319f76
# ╟─4f08fc6d-aa8e-4718-99e0-c8da07fe52dc
# ╠═2f9fdaa9-9183-4cd1-85e4-816aaa4b51ca
# ╟─53717c9c-475c-4374-a3fb-2f5459b2a5d3
# ╠═ce86e8af-fa5e-41da-9568-5379c0328b80
# ╟─f20ff624-d13f-4796-89d7-d13c7d87d4b9
# ╠═7c9d1e81-2a5c-4b5a-a625-53cb46011eea
# ╟─d267d626-589a-4f64-9362-28653ba1503f
# ╠═6866d623-70d1-49b8-a179-b9040a8a2ded
# ╟─9ee2a814-94b7-49ec-bcfc-29fe2767c7b3
# ╠═022b95d8-c013-4f18-945a-6df5fe917e3e
# ╟─86ebd244-1415-4acc-baaa-6d7a9ba9668f
# ╠═0dc372ca-0896-46e3-815b-965235c4512c
# ╟─4242c1fa-114b-4f14-a8ee-78fbbf472957
# ╠═f55696eb-3658-4296-82d6-93ba51d77052
# ╟─1a6bdcb9-6858-4f99-84c3-1347f78ef772
# ╠═53754be3-0d75-4365-a924-8810725870b9
# ╟─3b5d6394-6211-47e6-9357-bc69b2471aca
# ╠═33b686f4-020d-471b-91e4-b32a0da1bed6
# ╟─32feb06e-a7cb-4af9-9f6d-c66f089a1804
# ╠═36c15c2e-bb44-4d82-89bf-5b7d4bfb6da8
# ╟─3328cf9d-5382-4df0-8aff-271fa858c99d
# ╠═6fe94317-e4a5-42be-abc7-406735482c5c
# ╟─89cfa6e6-d0d1-4616-90a9-07e02efcb76a
# ╠═443d1273-3762-4631-85f0-d29e130d5245
# ╟─dfdba161-8775-42cf-8aa5-7c6129ca4620
# ╠═7c63277d-a47d-4ab3-aca9-221a1091f610
# ╟─6b50d54b-54d9-429c-94df-97f764c21559
# ╠═7d023b1c-4b49-4c7e-b755-5a785f1eac1b
# ╟─b77d0c93-e215-41a3-99fc-ad367c6db153
# ╟─9432dfff-4b44-4236-8b5d-f80fddc69d3a
# ╠═7aab7828-9a01-4393-a5bf-ee348b763112
# ╠═4c2c0635-9a57-4816-91b4-ccc5bdd39981
