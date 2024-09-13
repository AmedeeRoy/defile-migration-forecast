/* ERA5_LAND/HOURLY */
// https://developers.google.com/earth-engine/datasets/catalog/ECMWF_ERA5_LAND_HOURLY#bands
//var dataset = "ECMWF/ERA5_LAND/HOURLY"
//var selectors = ['temperature_2m', 'dewpoint_temperature_2m', 'total_precipitation', 'surface_pressure', 'u_component_of_wind_10m', 'v_component_of_wind_10m', 'surface_solar_radiation_downwards', 'surface_sensible_heat_flux'];

/* ERA5/HOURLY */
var dataset = "ECMWF/ERA5/HOURLY"
var selectors = ['temperature_2m', 'dewpoint_temperature_2m', 'total_precipitation', 'surface_pressure', 'u_component_of_wind_10m', 'v_component_of_wind_10m', 'v_component_of_wind_100m', 'u_component_of_wind_100m', 'instantaneous_10m_wind_gust', 'high_cloud_cover', 'low_cloud_cover', 'medium_cloud_cover', 'total_cloud_cover','surface_solar_radiation_downwards', 'convective_available_potential_energy'];

// [dewpoint_temperature_2m, temperature_2m, ice_temperature_layer_1, ice_temperature_layer_2, ice_temperature_layer_3, ice_temperature_layer_4, mean_sea_level_pressure, sea_surface_temperature, skin_temperature, surface_pressure, u_component_of_wind_100m, v_component_of_wind_100m, u_component_of_neutral_wind_10m, u_component_of_wind_10m, v_component_of_neutral_wind_10m, v_component_of_wind_10m, instantaneous_10m_wind_gust, mean_boundary_layer_dissipation, mean_convective_precipitation_rate, mean_convective_snowfall_rate, mean_eastward_gravity_wave_surface_stress, mean_eastward_turbulent_surface_stress, mean_evaporation_rate, mean_gravity_wave_dissipation, mean_large_scale_precipitation_fraction, mean_large_scale_precipitation_rate, mean_large_scale_snowfall_rate, mean_northward_gravity_wave_surface_stress, mean_northward_turbulent_surface_stress, mean_potential_evaporation_rate, mean_runoff_rate, mean_snow_evaporation_rate, mean_snowfall_rate, mean_snowmelt_rate, mean_sub_surface_runoff_rate, mean_surface_direct_short_wave_radiation_flux, mean_surface_direct_short_wave_radiation_flux_clear_sky, mean_surface_downward_long_wave_radiation_flux, mean_surface_downward_long_wave_radiation_flux_clear_sky, mean_surface_downward_short_wave_radiation_flux, mean_surface_downward_short_wave_radiation_flux_clear_sky, mean_surface_downward_uv_radiation_flux, mean_surface_latent_heat_flux, mean_surface_net_long_wave_radiation_flux, mean_surface_net_long_wave_radiation_flux_clear_sky, mean_surface_net_short_wave_radiation_flux, mean_surface_net_short_wave_radiation_flux_clear_sky, mean_surface_runoff_rate, mean_surface_sensible_heat_flux, mean_top_downward_short_wave_radiation_flux, mean_top_net_long_wave_radiation_flux, mean_top_net_long_wave_radiation_flux_clear_sky, mean_top_net_short_wave_radiation_flux, mean_top_net_short_wave_radiation_flux_clear_sky, mean_total_precipitation_rate, mean_vertically_integrated_moisture_divergence, clear_sky_direct_solar_radiation_at_surface, downward_uv_radiation_at_the_surface, forecast_logarithm_of_surface_roughness_for_heat, instantaneous_surface_sensible_heat_flux, near_ir_albedo_for_diffuse_radiation, near_ir_albedo_for_direct_radiation, surface_latent_heat_flux, surface_net_solar_radiation, surface_net_solar_radiation_clear_sky, surface_net_thermal_radiation, surface_net_thermal_radiation_clear_sky, surface_sensible_heat_flux, surface_solar_radiation_downward_clear_sky, surface_solar_radiation_downwards, surface_thermal_radiation_downward_clear_sky, surface_thermal_radiation_downwards, toa_incident_solar_radiation, top_net_solar_radiation, top_net_solar_radiation_clear_sky, top_net_thermal_radiation, top_net_thermal_radiation_clear_sky, total_sky_direct_solar_radiation_at_surface, uv_visible_albedo_for_diffuse_radiation, uv_visible_albedo_for_direct_radiation, cloud_base_height, high_cloud_cover, low_cloud_cover, medium_cloud_cover, total_cloud_cover, total_column_cloud_ice_water, total_column_cloud_liquid_water, lake_bottom_temperature, lake_cover, lake_depth, lake_ice_depth, lake_ice_temperature, lake_mix_layer_depth, lake_mix_layer_temperature, lake_shape_factor, lake_total_layer_temperature, evaporation, potential_evaporation, runoff, sub_surface_runoff, surface_runoff, convective_precipitation, convective_rain_rate, instantaneous_large_scale_surface_precipitation_fraction, large_scale_precipitation, large_scale_precipitation_fraction, large_scale_rain_rate, precipitation_type, total_column_rain_water, total_precipitation, convective_snowfall, convective_snowfall_rate_water_equivalent, large_scale_snowfall, large_scale_snowfall_rate_water_equivalent, snow_albedo, snow_density, snow_depth, snow_evaporation, snowfall, snowmelt, temperature_of_snow_layer, total_column_snow_water, soil_temperature_level_1, soil_temperature_level_2, soil_temperature_level_3, soil_temperature_level_4, soil_type, vertical_integral_of_divergence_of_cloud_frozen_water_flux, vertical_integral_of_divergence_of_cloud_liquid_water_flux, vertical_integral_of_divergence_of_geopotential_flux, vertical_integral_of_divergence_of_kinetic_energy_flux, vertical_integral_of_divergence_of_mass_flux, vertical_integral_of_divergence_of_moisture_flux, vertical_integral_of_divergence_of_ozone_flux, vertical_integral_of_divergence_of_thermal_energy_flux, vertical_integral_of_divergence_of_total_energy_flux, vertical_integral_of_eastward_cloud_frozen_water_flux, vertical_integral_of_eastward_cloud_liquid_water_flux, vertical_integral_of_eastward_geopotential_flux, vertical_integral_of_eastward_heat_flux, vertical_integral_of_eastward_kinetic_energy_flux, vertical_integral_of_eastward_mass_flux, vertical_integral_of_eastward_ozone_flux, vertical_integral_of_eastward_total_energy_flux, vertical_integral_of_eastward_water_vapour_flux, vertical_integral_of_energy_conversion, vertical_integral_of_kinetic_energy, vertical_integral_of_mass_of_atmosphere, vertical_integral_of_mass_tendency, vertical_integral_of_northward_cloud_frozen_water_flux, vertical_integral_of_northward_cloud_liquid_water_flux, vertical_integral_of_northward_geopotential_flux, vertical_integral_of_northward_heat_flux, vertical_integral_of_northward_kinetic_energy_flux, vertical_integral_of_northward_mass_flux, vertical_integral_of_northward_ozone_flux, high_vegetation_cover, leaf_area_index_high_vegetation, leaf_area_index_low_vegetation, low_vegetation_cover, type_of_high_vegetation, type_of_low_vegetation, air_density_over_the_oceans, coefficient_of_drag_with_waves, free_convective_velocity_over_the_oceans, maximum_individual_wave_height, mean_direction_of_total_swell, mean_direction_of_wind_waves, mean_period_of_total_swell, mean_period_of_wind_waves, mean_square_slope_of_waves, mean_wave_direction, mean_wave_direction_of_first_swell_partition, mean_wave_direction_of_second_swell_partition, mean_wave_direction_of_third_swell_partition, mean_wave_period, mean_wave_period_based_on_first_moment, mean_wave_period_based_on_first_moment_for_swell, mean_wave_period_based_on_first_moment_for_wind_waves, mean_wave_period_based_on_second_moment_for_swell, mean_wave_period_based_on_second_moment_for_wind_waves, mean_wave_period_of_first_swell_partition, mean_wave_period_of_second_swell_partition, mean_wave_period_of_third_swell_partition, mean_zero_crossing_wave_period, model_bathymetry, normalized_energy_flux_into_ocean, normalized_energy_flux_into_waves, normalized_stress_into_ocean, ocean_surface_stress_equivalent_10m_neutral_wind_direction, ocean_surface_stress_equivalent_10m_neutral_wind_speed, peak_wave_period, period_corresponding_to_maximum_individual_wave_height, significant_height_of_combined_wind_waves_and_swell, significant_height_of_total_swell, significant_height_of_wind_waves, significant_wave_height_of_first_swell_partition, significant_wave_height_of_second_swell_partition, significant_wave_height_of_third_swell_partition, angle_of_sub_gridscale_orography, anisotropy_of_sub_gridscale_orography, benjamin_feir_index, boundary_layer_dissipation, boundary_layer_height, charnock, convective_available_potential_energy, convective_inhibition, duct_base_height, eastward_gravity_wave_surface_stress, eastward_turbulent_surface_stress, forecast_albedo, forecast_surface_roughness, friction_velocity, gravity_wave_dissipation, instantaneous_eastward_turbulent_surface_stress, instantaneous_moisture_flux, instantaneous_northward_turbulent_surface_stress, k_index, land_sea_mask, mean_vertical_gradient_of_refractivity_inside_trapping_layer, minimum_vertical_gradient_of_refractivity_inside_trapping_layer, northward_gravity_wave_surface_stress, northward_turbulent_surface_stress, sea_ice_cover, skin_reservoir_content, slope_of_sub_gridscale_orography, standard_deviation_of_filtered_subgrid_orography, standard_deviation_of_orography, total_column_ozone, total_column_supercooled_liquid_water, total_column_water, total_column_water_vapour, total_totals_index, trapping_layer_base_height, trapping_layer_top_height, u_component_stokes_drift, v_component_stokes_drift, vertical_integral_of_northward_total_energy_flux, vertical_integral_of_northward_water_vapour_flux, vertical_integral_of_potential_and_internal_energy, vertical_integral_of_potential_internal_and_latent_energy, vertical_integral_of_temperature, vertical_integral_of_thermal_energy, vertical_integral_of_total_energy, vertically_integrated_moisture_divergence, volumetric_soil_water_layer_1, volumetric_soil_water_layer_2, volumetric_soil_water_layer_3, volumetric_soil_water_layer_4, wave_spectral_directional_width, wave_spectral_directional_width_for_swell, wave_spectral_directional_width_for_wind_waves, wave_spectral_kurtosis, wave_spectral_peakedness, wave_spectral_skewness, zero_degree_level, wind_gust_since_previous_post_processing_10m, geopotential, maximum_2m_temperature_since_previous_post_processing, maximum_total_precipitation_rate_since_previous_post_processing, minimum_2m_temperature_since_previous_post_processing, minimum_total_precipitation_rate_since_previous_post_processing, divergence_500hPa, divergence_850hPa, fraction_of_cloud_cover_500hPa, fraction_of_cloud_cover_850hPa, ozone_mass_mixing_ratio_500hPa, ozone_mass_mixing_ratio_850hPa, potential_vorticity_500hPa, potential_vorticity_850hPa, relative_humidity_500hPa, relative_humidity_850hPa, specific_cloud_ice_water_content_500hPa, specific_cloud_ice_water_content_850hPa, specific_cloud_liquid_water_content_500hPa, specific_cloud_liquid_water_content_850hPa, specific_humidity_500hPa, specific_humidity_850hPa, specific_rain_water_content_500hPa, specific_rain_water_content_850hPa, specific_snow_water_content_500hPa, specific_snow_water_content_850hPa, temperature_500hPa, temperature_850hPa, u_component_of_wind_500hPa, u_component_of_wind_850hPa, v_component_of_wind_500hPa, v_component_of_wind_850hPa, vertical_velocity_500hPa, vertical_velocity_850hPa, vorticity_500hPa, vorticity_850hPa].


/* ERA5_LAND/DAILY_AGGR */
// https://developers.google.com/earth-engine/datasets/catalog/ECMWF_ERA5_LAND_DAILY_AGGR#bands
//var selectors = ['temperature_2m', 'total_precipitation_sum', 'surface_pressure', 'u_component_of_wind_10m', 'v_component_of_wind_10m']
//var dataset = "ECMWF/ERA5_LAND/DAILY_AGGR";

/* MODIS */
// var selectors = 'NDVI'
// var dataset = 'MODIS/061/MOD13A2';


/*
var name = "Mwamba";
var position = ee.Geometry.Point([39.99, -3.34]); //Mwamba
var name = "Mtwari";
var position = ee.Geometry.Point([40, -11]); 
var name = "defile";
var position = ee.Geometry.Point([5.914877, 46.117215]); // defile de lecluse
*/

// Main location
//var locations = {"Defile": ee.Geometry.Point([5.914877, 46.117215])}

// Locations for hourly
var locations = {
    "Defile": ee.Geometry.Point([5.914877, 46.117215]),
    "Basel": ee.Geometry.Point([ 7.603047,47.561227]),
    "MontTendre": ee.Geometry.Point([ 6.309533, 46.594542]),
    "Chasseral": ee.Geometry.Point([ 7.058849, 47.132963]),
    "ColGrandSaintBernard": ee.Geometry.Point([ 7.165453,45.868846]),
    "Schaffhausen": ee.Geometry.Point([ 8.634737,47.697376]),
    "Dijon": ee.Geometry.Point([ 5.033933, 47.323463])
};
/*
//Location for daily
var locations = {
    "Munich": ee.Geometry.Point([ 11.572036,48.144714]),
    "Frankfurt": ee.Geometry.Point([ 8.670885, 50.118894]),
    "Stuttgart": ee.Geometry.Point([ 9.181457, 48.775951]),
    "Berlin": ee.Geometry.Point([ 13.404231, 52.522825]),
    "Prague": ee.Geometry.Point([ 14.436136, 50.072197]),
    "Warsaw": ee.Geometry.Point([ 21.014928, 52.216390])
};

// Location really far
var locations = {
    "Berlin": ee.Geometry.Point([ 13.404231, 52.522825]),
    "Prague": ee.Geometry.Point([ 14.436136, 50.072197]),
    "Warsaw": ee.Geometry.Point([ 21.014928, 52.216390])
};
*/


//Map.addLayer(ee.ImageCollection(dataset).first().mask(), {opacity:0.5});
Map.addLayer(ee.FeatureCollection(Object.keys(locations).map(function(name) {
  return ee.Feature(locations[name], {name: name});
})), {color: 'red'}, 'Markers');

var start = "1966-01-01";
var end = "2024-01-01";

var era5_filtered = ee.ImageCollection(dataset).select(selectors).filter(ee.Filter.date(start, end))


for (var name in locations){
  var position = locations[name];
  var era5_export = era5_filtered.map(function (im) { 
    return ee.Feature(null,im.reduceRegion(ee.Reducer.first(), position,1000)).set("datetime", im.date().format()); });
  
  Export.table.toDrive({
      collection: era5_export,
      fileNamePrefix: name,
      selectors: ['datetime'].concat(selectors),
  })
}